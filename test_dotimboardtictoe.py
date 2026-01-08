#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import json
import uuid
import urllib.request
import urllib.error
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file
from io import BytesIO

# =========================
# CONFIG
# =========================
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))
CAM_DEV = os.environ.get("CAM_DEV", "0")
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))

# camera đang bị ngược => mặc định TRUE
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

# board 4x6 (cols=4, rows=6)
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))
GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))

# endpoint nhận ảnh composite để GPT/server xác nhận nước cờ
SCAN_API_URL = os.environ.get(
    "SCAN_API_URL",
    "https://embeddedprogramming-healtheworldserver.up.railway.app/scan_chess",
)

# mở rộng quad bàn cờ thêm chút (thay “2cm” vì không có scale => ratio)
BOARD_PAD_RATIO = float(os.environ.get("BOARD_PAD_RATIO", "0.06"))  # 0.04~0.10 tùy bạn

# marker vàng HSV threshold (bạn có thể tinh chỉnh)
Y_H_LO = int(os.environ.get("Y_H_LO", "18"))
Y_S_LO = int(os.environ.get("Y_S_LO", "70"))
Y_V_LO = int(os.environ.get("Y_V_LO", "70"))
Y_H_HI = int(os.environ.get("Y_H_HI", "40"))
Y_S_HI = int(os.environ.get("Y_S_HI", "255"))
Y_V_HI = int(os.environ.get("Y_V_HI", "255"))

# lọc marker theo diện tích (px) - tùy camera
MARKER_AREA_MIN = int(os.environ.get("MARKER_AREA_MIN", "120"))
MARKER_AREA_MAX = int(os.environ.get("MARKER_AREA_MAX", "20000"))

# warp size: cell size px
CELL_SIZE = int(os.environ.get("CELL_SIZE", "140"))  # 120-180 tùy bạn
WARP_W = GRID_COLS * CELL_SIZE
WARP_H = GRID_ROWS * CELL_SIZE

# edges params
CANNY1 = int(os.environ.get("CANNY1", "60"))
CANNY2 = int(os.environ.get("CANNY2", "160"))

# morph params
MORPH_K = int(os.environ.get("MORPH_K", "3"))

# giảm giật: chỉ poll state chậm
STATE_POLL_MS = int(os.environ.get("STATE_POLL_MS", "800"))

# =========================
# HTTP upload helper
# =========================
def _encode_jpeg(img_bgr, quality=JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return b""
    return buf.tobytes()


def _post_image_to_api(image_bytes: bytes, timeout_sec: int = 25) -> Optional[Dict[str, Any]]:
    """POST multipart image -> SCAN_API_URL, expecting JSON (optional)."""
    if not image_bytes:
        return None

    boundary = uuid.uuid4().hex
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + image_bytes + footer

    req = urllib.request.Request(
        SCAN_API_URL,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        print(f"[API] HTTPError {exc.code}: {detail[:250]}", flush=True)
        return None
    except urllib.error.URLError as exc:
        print(f"[API] URLError: {exc}", flush=True)
        return None
    except Exception as exc:
        print(f"[API] parse failed: {exc}", flush=True)
        return None


# =========================
# Geometry helpers
# =========================
def order_points_tl_tr_bl_br(pts: np.ndarray) -> np.ndarray:
    """pts: (4,2) -> ordered: TL, TR, BL, BR"""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, bl, br], dtype=np.float32)


def pad_quad(quad: np.ndarray, pad_ratio: float) -> np.ndarray:
    """
    Mở rộng quad quanh tâm.
    quad: (4,2) ordered or not (we'll treat as 4 points).
    """
    q = np.array(quad, dtype=np.float32).reshape(4, 2)
    c = q.mean(axis=0)
    v = q - c
    q2 = c + v * (1.0 + float(max(0.0, pad_ratio)))
    return q2


def infer_4th_point_from_3(pts3: np.ndarray) -> Optional[np.ndarray]:
    """
    Given 3 corners of a parallelogram/rectangle, infer 4th.
    Robust: choose the pair with largest distance as diagonal (A,B),
    third point is C, then D = A + B - C.
    """
    pts = np.array(pts3, dtype=np.float32).reshape(3, 2)
    # pairwise distances
    d01 = np.linalg.norm(pts[0] - pts[1])
    d02 = np.linalg.norm(pts[0] - pts[2])
    d12 = np.linalg.norm(pts[1] - pts[2])

    if d01 >= d02 and d01 >= d12:
        A, B, C = pts[0], pts[1], pts[2]
    elif d02 >= d01 and d02 >= d12:
        A, B, C = pts[0], pts[2], pts[1]
    else:
        A, B, C = pts[1], pts[2], pts[0]

    D = A + B - C
    return np.array([A, B, C, D], dtype=np.float32)


def clamp_points_to_image(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    p = np.array(pts, dtype=np.float32).reshape(-1, 2)
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    return p


# =========================
# Vision: marker detect
# =========================
def detect_yellow_markers(frame_bgr: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Return list of marker centers [(x,y)], plus mask and debug image.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([Y_H_LO, Y_S_LO, Y_V_LO], dtype=np.uint8)
    hi = np.array([Y_H_HI, Y_S_HI, Y_V_HI], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    # clean noise
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    dbg = frame_bgr.copy()

    for c in cnts:
        area = cv2.contourArea(c)
        if area < MARKER_AREA_MIN or area > MARKER_AREA_MAX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 2 or h <= 2:
            continue
        ar = w / float(h + 1e-6)
        # marker là hình chữ nhật nhỏ, cho phép hơi dài
        if ar < 0.3 or ar > 3.5:
            continue

        M = cv2.moments(c)
        if abs(M.get("m00", 0.0)) < 1e-6:
            continue
        cx = int(M["m10"] / (M["m00"] + 1e-6))
        cy = int(M["m01"] / (M["m00"] + 1e-6))
        centers.append((cx, cy))

        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(dbg, (cx, cy), 5, (0, 0, 255), -1)

    # nếu nhiều hơn 4 -> lấy 4 điểm xa nhau nhất theo convex hull
    if len(centers) > 4:
        pts = np.array(centers, dtype=np.int32)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        # nếu hull > 4, lấy 4 điểm theo extreme corners (sum/diff)
        if hull.shape[0] >= 4:
            s = hull.sum(axis=1)
            d = np.diff(hull, axis=1).reshape(-1)
            tl = hull[np.argmin(s)]
            br = hull[np.argmax(s)]
            tr = hull[np.argmin(d)]
            bl = hull[np.argmax(d)]
            centers = [(int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])),
                       (int(bl[0]), int(bl[1])), (int(br[0]), int(br[1]))]
        else:
            centers = centers[:4]

    return centers, mask, dbg


# =========================
# Vision: processing stages
# =========================
def stage_gray_blur(frame_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(g, (7, 7), 0)
    return b


def stage_edges(gray_blur: np.ndarray) -> np.ndarray:
    e = cv2.Canny(gray_blur, CANNY1, CANNY2)
    return e


def stage_edges_closed(edges: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    # close để nối nét
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    # thêm 1 lần dilate nhẹ
    closed = cv2.dilate(closed, k, iterations=1)
    return closed


def find_board_quad_from_edges(edges_closed: np.ndarray, min_area_ratio: float = 0.08) -> Optional[np.ndarray]:
    """
    Tìm 1 contour lớn nhất dạng tứ giác từ ảnh edges/closed.
    (dùng khi marker fail)
    """
    h, w = edges_closed.shape[:2]
    min_area = float(h * w) * float(min_area_ratio)

    cnts, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)

    if best is None:
        return None
    return np.array(best, dtype=np.float32)


def compute_angle_from_quad(quad_ordered: np.ndarray) -> float:
    """
    Góc nghiêng dựa trên cạnh top (TL->TR)
    """
    tl, tr = quad_ordered[0], quad_ordered[1]
    dx = float(tr[0] - tl[0])
    dy = float(tr[1] - tl[1])
    ang = np.degrees(np.arctan2(dy, dx))
    return float(ang)


def warp_board(frame_bgr: np.ndarray, quad_ordered: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warp board to WARP_W x WARP_H
    Return: warped_bgr, H, Hinv
    """
    src = quad_ordered.astype(np.float32)
    dst = np.array([
        [0, 0],
        [WARP_W - 1, 0],
        [0, WARP_H - 1],
        [WARP_W - 1, WARP_H - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    Hinv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(frame_bgr, H, (WARP_W, WARP_H))
    return warped, H, Hinv


def draw_quad(frame_bgr: np.ndarray, quad_ordered: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
    out = frame_bgr.copy()
    pts = quad_ordered.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
    return out


def grid_cell_quads_in_original(Hinv: np.ndarray) -> List[List[np.ndarray]]:
    """
    Tạo quad cho mỗi cell trong hệ warp, rồi map ngược về ảnh gốc.
    Return: cell_quads[row][col] = (4,2) float32 in original image space
    """
    cell_quads = []
    for r in range(GRID_ROWS):
        row_list = []
        for c in range(GRID_COLS):
            x0 = c * CELL_SIZE
            y0 = r * CELL_SIZE
            x1 = (c + 1) * CELL_SIZE
            y1 = (r + 1) * CELL_SIZE

            pts_warp = np.array([
                [x0, y0],
                [x1, y0],
                [x0, y1],
                [x1, y1],
            ], dtype=np.float32).reshape(-1, 1, 2)

            pts_org = cv2.perspectiveTransform(pts_warp, Hinv).reshape(4, 2)
            row_list.append(pts_org.astype(np.float32))
        cell_quads.append(row_list)
    return cell_quads


def make_cells_composite(warped_bgr: np.ndarray) -> np.ndarray:
    """
    Cắt tất cả ô từ warped, làm đậm viền, ghép thành 1 ảnh lớn để gửi endpoint.
    """
    tiles = []
    for r in range(GRID_ROWS):
        row_tiles = []
        for c in range(GRID_COLS):
            x0 = c * CELL_SIZE
            y0 = r * CELL_SIZE
            x1 = (c + 1) * CELL_SIZE
            y1 = (r + 1) * CELL_SIZE
            crop = warped_bgr[y0:y1, x0:x1].copy()

            # làm đậm đường viền/nét: edges -> overlay
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            e = cv2.Canny(g, 60, 170)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            e = cv2.dilate(e, k, iterations=1)
            e3 = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)

            # invert edges to black lines on white-ish background: tùy bạn, ở đây giữ trắng trên đen
            # trộn nhẹ để GPT thấy rõ nét
            combo = cv2.addWeighted(crop, 0.80, e3, 0.20, 0)

            # label r,c lên tile
            cv2.putText(combo, f"({r},{c})", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            row_tiles.append(combo)
        tiles.append(np.hstack(row_tiles))
    composite = np.vstack(tiles)
    return composite


# =========================
# State store
# =========================
class BoardState:
    def __init__(self, rows: int, cols: int):
        self._lock = threading.Lock()
        self.rows = int(rows)
        self.cols = int(cols)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self._cells_text = "-"
        self._scan_status = "idle"
        self._angle = None
        self._scan_id = 0

        # store last stages as jpeg bytes (only updated after scan)
        self._stages: Dict[str, bytes] = {}

    def set_scan(self, status: str, angle: Optional[float]):
        with self._lock:
            self._scan_status = str(status)
            self._angle = angle

    def set_board(self, board: List[List[int]]):
        with self._lock:
            self._board = [row[:] for row in board]

    def set_cells_text(self, text: str):
        with self._lock:
            self._cells_text = text

    def bump_scan_id(self):
        with self._lock:
            self._scan_id += 1

    def set_stage(self, name: str, img_bgr_or_gray: np.ndarray):
        # store as jpeg
        if img_bgr_or_gray is None:
            return
        if len(img_bgr_or_gray.shape) == 2:
            img = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2BGR)
        else:
            img = img_bgr_or_gray
        self._stages[name] = _encode_jpeg(img, quality=80)

    def clear_stages(self):
        with self._lock:
            self._stages = {}

    def get_stage(self, name: str) -> Optional[bytes]:
        with self._lock:
            return self._stages.get(name)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            board = [row[:] for row in self._board]
            scan_status = self._scan_status
            angle = self._angle
            scan_id = self._scan_id
            cells_text = self._cells_text

        empty = sum(1 for r in board for v in r if v == 0)
        player = sum(1 for r in board for v in r if v == 1)
        robot = sum(1 for r in board for v in r if v == 2)

        return {
            "rows": self.rows,
            "cols": self.cols,
            "scan_status": scan_status,
            "angle": angle,
            "empty": empty,
            "player": player,
            "robot": robot,
            "board": board,
            "cells_text": cells_text,
            "scan_id": scan_id,
        }


# =========================
# Web + Camera
# =========================
class CameraWeb:
    def __init__(self, state: BoardState):
        self.state = state
        self.app = Flask("scan_board")
        self._lock = threading.Lock()
        self._last_frame = None

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._play_requested = threading.Event()

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

        @self.app.get("/state.json")
        def state_json():
            return jsonify(self.state.snapshot())

        @self.app.get("/play")
        def play():
            self._play_requested.set()
            return jsonify({"ok": True})

        @self.app.get("/mjpeg")
        def mjpeg():
            return Response(self._mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.get("/stage/<name>.jpg")
        def stage(name):
            b = self.state.get_stage(name)
            if not b:
                return ("", 404)
            return send_file(BytesIO(b), mimetype="image/jpeg")

    def consume_scan_request(self) -> bool:
        if self._play_requested.is_set():
            self._play_requested.clear()
            return True
        return False

    def get_last_frame(self):
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def _capture_loop(self):
        dev = int(CAM_DEV) if str(CAM_DEV).isdigit() else CAM_DEV
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        if not cap.isOpened():
            print(f"[CAM] cannot open camera: {dev}", flush=True)
            self._failed.set()
            self._ready.set()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            with self._lock:
                self._last_frame = frame
            self._ready.set()
            time.sleep(0.01)

        cap.release()

    def _mjpeg_gen(self):
        # live view; không nháy vì stages chỉ update sau scan
        while not self._stop.is_set():
            frame = self.get_last_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                time.sleep(0.03)
                continue

            jpg = buf.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.05)

    def _html(self) -> str:
        # tránh format() với JS braces => dùng placeholder
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Board 4x6</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }
    .wrap { display:flex; gap:16px; padding:16px; align-items:flex-start; }
    .left { width:340px; }
    .mid { flex:1; min-width:520px; }
    .right { width:420px; }
    .card { background:#111827; border:1px solid #223; border-radius:12px; padding:12px; }
    .kv { margin:8px 0; }
    .k { color:#93c5fd; }
    .cells { font-size:12px; color:#cbd5f5; white-space:pre; max-height:420px; overflow:auto; background:#0b1220; border-radius:10px; padding:10px; border:1px solid #223; }
    .btn { background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:8px 12px; border-radius:10px; cursor:pointer; }
    .video { border:1px solid #223; border-radius:12px; width:__CAM_W__px; height:__CAM_H__px; background:#000; }
    .stage { margin-top:12px; }
    .stage h4 { margin:10px 0 6px 0; font-size:13px; color:#93c5fd; }
    .stage img { width:100%; border:1px solid #223; border-radius:12px; background:#000; }
    .muted { color:#94a3b8; font-size:12px; }
  </style>
</head>
<body>
  <div class="wrap">

    <div class="left">
      <div class="card">
        <div class="kv"><span class="k">Board:</span> rows=<span id="rows">-</span>, cols=<span id="cols">-</span></div>
        <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">idle</span></div>
        <div class="kv"><span class="k">Angle:</span> <span id="angle">-</span></div>

        <div class="kv"><span class="k">Empty:</span> <span id="empty">-</span></div>
        <div class="kv"><span class="k">Player X:</span> <span id="player">-</span></div>
        <div class="kv"><span class="k">Robot O:</span> <span id="robot">-</span></div>

        <div class="kv"><span class="k">Board state:</span></div>
        <div id="board" class="cells">-</div>

        <div class="kv"><span class="k">Cells (server):</span></div>
        <div id="cells_text" class="cells">-</div>

        <div style="margin-top:10px;">
          <button class="btn" onclick="playScan()">Play Scan</button>
          <div class="muted" style="margin-top:8px;">Tip: Web chỉ cập nhật ảnh stages sau khi scan xong để tránh giật.</div>
        </div>
      </div>
    </div>

    <div class="mid">
      <img class="video" id="cam" src="/mjpeg" />
    </div>

    <div class="right">
      <div class="card">
        <div class="kv"><span class="k">Processing Stages</span> <span class="muted">(update theo scan_id)</span></div>

        <div class="stage"><h4>1_marker_mask</h4><img id="s_marker_mask" src="" /></div>
        <div class="stage"><h4>2_marker_debug</h4><img id="s_marker_dbg" src="" /></div>
        <div class="stage"><h4>3_edges_closed</h4><img id="s_edges_closed" src="" /></div>
        <div class="stage"><h4>4_board_quad</h4><img id="s_board_quad" src="" /></div>
        <div class="stage"><h4>5_warp</h4><img id="s_warp" src="" /></div>
        <div class="stage"><h4>6_cells_composite_sent</h4><img id="s_cells_comp" src="" /></div>
      </div>
    </div>

  </div>

<script>
let lastScanId = -1;

function formatBoard(board) {
  if (!board || !board.length) return '-';
  return board.map(row => row.join(' ')).join('\\n');
}

async function tick() {
  try {
    const r = await fetch('/state.json', {cache:'no-store'});
    const js = await r.json();

    document.getElementById('rows').textContent = js.rows ?? '-';
    document.getElementById('cols').textContent = js.cols ?? '-';
    document.getElementById('scan_status').textContent = js.scan_status ?? '-';
    document.getElementById('angle').textContent = (js.angle === null || js.angle === undefined) ? '-' : js.angle.toFixed(2);

    document.getElementById('empty').textContent = js.empty ?? '-';
    document.getElementById('player').textContent = js.player ?? '-';
    document.getElementById('robot').textContent = js.robot ?? '-';

    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('cells_text').textContent = js.cells_text ?? '-';

    const scanId = js.scan_id ?? 0;
    if (scanId !== lastScanId) {
      lastScanId = scanId;
      // chỉ reload stages khi scan_id đổi => không chớp liên tục
      setStage('s_marker_mask',  '/stage/marker_mask.jpg?sid=' + scanId);
      setStage('s_marker_dbg',   '/stage/marker_debug.jpg?sid=' + scanId);
      setStage('s_edges_closed', '/stage/edges_closed.jpg?sid=' + scanId);
      setStage('s_board_quad',   '/stage/board_quad.jpg?sid=' + scanId);
      setStage('s_warp',         '/stage/warp.jpg?sid=' + scanId);
      setStage('s_cells_comp',   '/stage/cells_composite.jpg?sid=' + scanId);
    }
  } catch(e) {}
}

function setStage(id, url) {
  const el = document.getElementById(id);
  if (!el) return;
  el.src = url;
}

async function playScan() {
  try { await fetch('/play', {cache:'no-store'}); } catch(e) {}
  tick();
}

setInterval(tick, __POLL_MS__);
tick();
</script>
</body>
</html>
"""
        html = html.replace("__CAM_W__", str(CAM_W)).replace("__CAM_H__", str(CAM_H))
        html = html.replace("__POLL_MS__", str(STATE_POLL_MS))
        return html

    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True).start()

        threading.Thread(
            target=lambda: self.app.run(
                host="0.0.0.0",
                port=WEB_PORT,
                debug=False,
                use_reloader=False,
                threaded=True,
            ),
            daemon=True,
        ).start()

        print(f"[WEB] http://<pi_ip>:{WEB_PORT}/", flush=True)

    def wait_ready(self, timeout_sec: float = 5.0) -> bool:
        self._ready.wait(timeout=max(0.1, float(timeout_sec)))
        return self._ready.is_set() and not self._failed.is_set()


# =========================
# Board logic helpers
# =========================
def empty_board(rows: int, cols: int) -> List[List[int]]:
    return [[0 for _ in range(cols)] for _ in range(rows)]


def board_from_server_cells(rows: int, cols: int, result: Dict[str, Any]) -> Tuple[List[List[int]], str]:
    """
    Nếu endpoint trả cells kiểu:
      cells: [{row, col, state}]
    state: player_x / robot_o / empty ...
    """
    board = empty_board(rows, cols)
    lines = []
    cells = result.get("cells", []) if isinstance(result, dict) else []
    for cell in cells:
        try:
            r = int(cell.get("row", 0))
            c = int(cell.get("col", 0))
            st = str(cell.get("state", "empty"))
        except Exception:
            continue
        if 0 <= r < rows and 0 <= c < cols:
            if st == "player_x":
                board[r][c] = 1
            elif st in ("robot_o", "robot_line", "robot"):
                board[r][c] = 2
            else:
                board[r][c] = 0
        lines.append(f"({r},{c}) {st}")
    return board, "\n".join(lines) if lines else "-"


def overlay_cells_quads(frame_bgr: np.ndarray, cell_quads: List[List[np.ndarray]], board: List[List[int]]) -> np.ndarray:
    """
    Vẽ quad từng cell theo đúng phối cảnh (project ngược từ warp).
    """
    out = frame_bgr.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            q = cell_quads[r][c].astype(np.int32).reshape(-1, 1, 2)
            val = board[r][c]
            if val == 1:
                color = (0, 255, 255)   # X
                thickness = 2
            elif val == 2:
                color = (255, 0, 255)   # O
                thickness = 2
            else:
                color = (60, 60, 90)
                thickness = 1
            cv2.polylines(out, [q], True, color, thickness, cv2.LINE_AA)
    return out


# =========================
# Main scan pipeline
# =========================
def run_scan_pipeline(frame_bgr: np.ndarray, state: BoardState) -> Dict[str, Any]:
    """
    Pipeline:
    - marker detect (4 or 3 -> infer 4th)
    - fallback: find quad from edges
    - pad quad
    - warp
    - edges/closed in warp (for debug)
    - composite cells + send to endpoint
    - overlay quad + cell quads onto original
    """
    h, w = frame_bgr.shape[:2]
    state.clear_stages()

    # 1) marker detect
    centers, marker_mask, marker_dbg = detect_yellow_markers(frame_bgr)
    state.set_stage("marker_mask", marker_mask)
    state.set_stage("marker_debug", marker_dbg)

    quad = None
    quad_src = "marker"

    if len(centers) == 4:
        quad = np.array(centers, dtype=np.float32)
    elif len(centers) == 3:
        quad4 = infer_4th_point_from_3(np.array(centers, dtype=np.float32))
        if quad4 is not None:
            quad = quad4.reshape(4, 2)
    else:
        quad = None

    # 2) edges pipeline for fallback + debug
    gb = stage_gray_blur(frame_bgr)
    ed = stage_edges(gb)
    edc = stage_edges_closed(ed)
    state.set_stage("edges_closed", edc)

    if quad is None:
        # fallback: find quad from edges
        q2 = find_board_quad_from_edges(edc, min_area_ratio=0.06)
        if q2 is not None:
            quad = q2
            quad_src = "edges"

    if quad is None:
        # fail
        fail_img = frame_bgr.copy()
        cv2.putText(fail_img, "board quad NOT found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv2.LINE_AA)
        state.set_stage("board_quad", fail_img)
        return {"ok": False, "reason": "quad_not_found"}

    # 3) order + pad
    quad = clamp_points_to_image(quad, w, h)
    quad_ord = order_points_tl_tr_bl_br(quad)
    quad_pad = pad_quad(quad_ord, BOARD_PAD_RATIO)
    quad_pad = clamp_points_to_image(quad_pad, w, h)
    quad_pad = order_points_tl_tr_bl_br(quad_pad)

    angle = compute_angle_from_quad(quad_pad)

    # 4) warp
    warped, H, Hinv = warp_board(frame_bgr, quad_pad)

    # show quad on original
    board_quad_img = draw_quad(frame_bgr, quad_pad, color=(0,255,255), thickness=3)
    cv2.putText(board_quad_img, f"angle={angle:.1f} src={quad_src}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2, cv2.LINE_AA)
    state.set_stage("board_quad", board_quad_img)
    state.set_stage("warp", warped)

    # 5) build composite cells image (to send GPT/server)
    cells_comp = make_cells_composite(warped)
    state.set_stage("cells_composite", cells_comp)

    # 6) send to endpoint
    api_result = _post_image_to_api(_encode_jpeg(cells_comp, quality=80))
    board = empty_board(GRID_ROWS, GRID_COLS)
    cells_text = "-"

    if isinstance(api_result, dict) and (api_result.get("cells") is not None):
        board, cells_text = board_from_server_cells(GRID_ROWS, GRID_COLS, api_result)
    else:
        # nếu server không trả cells, vẫn để board empty
        if api_result is None:
            cells_text = "[API] no response / parse fail"
        else:
            cells_text = "[API] returned but no 'cells' field"

    # 7) overlay cell quads (project ngược)
    cell_quads = grid_cell_quads_in_original(Hinv)
    overlay_img = overlay_cells_quads(board_quad_img, cell_quads, board)

    # update stage: board_quad now includes cell quads
    state.set_stage("board_quad", overlay_img)

    return {"ok": True, "angle": angle, "board": board, "cells_text": cells_text}


# =========================
# Main loop
# =========================
def main():
    print("[START] tictactoe_scan_4x6", flush=True)
    print(f"[CFG] SCAN_API_URL={SCAN_API_URL}", flush=True)
    print(f"[CFG] rows={GRID_ROWS} cols={GRID_COLS} warp={WARP_W}x{WARP_H} cell={CELL_SIZE}", flush=True)
    print(f"[CFG] ROTATE_180={ROTATE_180}", flush=True)
    print(f"[CFG] BOARD_PAD_RATIO={BOARD_PAD_RATIO}", flush=True)

    state = BoardState(rows=GRID_ROWS, cols=GRID_COLS)
    cam = CameraWeb(state)
    cam.start()

    if not cam.wait_ready(timeout_sec=6.0):
        print("[CAM] not ready, stop", flush=True)
        return

    print("[WEB] press Play Scan to run pipeline", flush=True)

    try:
        while True:
            if cam.consume_scan_request():
                # reset visible status immediately
                state.set_scan("scanning", None)
                state.set_board(empty_board(GRID_ROWS, GRID_COLS))
                state.set_cells_text("-")

                frame = cam.get_last_frame()
                if frame is None:
                    state.set_scan("failed", None)
                    state.bump_scan_id()
                    continue

                res = run_scan_pipeline(frame, state)
                if not res.get("ok"):
                    state.set_scan("failed", None)
                else:
                    state.set_scan("ready", float(res.get("angle", 0.0)))
                    state.set_board(res.get("board", empty_board(GRID_ROWS, GRID_COLS)))
                    state.set_cells_text(res.get("cells_text", "-"))

                # IMPORTANT: bump scan id at end so web reload stages exactly once
                state.bump_scan_id()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
