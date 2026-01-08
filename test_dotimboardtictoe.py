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
from flask import Flask, Response, jsonify

# =========================
# CONFIG
# =========================
WEB_PORT = 8000
CAM_DEV = os.environ.get("CAM_DEV", "0")
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))

# Force rotate 180 by default
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

# Board size: 4x6
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))
GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))

SCAN_API_URL = os.environ.get(
    "SCAN_API_URL",
    "https://embeddedprogramming-healtheworldserver.up.railway.app/scan_chess",
)

# Warp output size
WARP_SIZE = int(os.environ.get("WARP_SIZE", "520"))

# Highlight alpha
HILITE_ALPHA = float(os.environ.get("HILITE_ALPHA", "0.28"))

# Circle radius filter relative to cell size (for local O detection preview)
MIN_R_RATIO = float(os.environ.get("MIN_R_RATIO", "0.12"))
MAX_R_RATIO = float(os.environ.get("MAX_R_RATIO", "0.30"))

# HSV blue range (tuned for blue marker tape/pen)
BLUE_H_LO = int(os.environ.get("BLUE_H_LO", "85"))
BLUE_H_HI = int(os.environ.get("BLUE_H_HI", "140"))
BLUE_S_LO = int(os.environ.get("BLUE_S_LO", "55"))
BLUE_V_LO = int(os.environ.get("BLUE_V_LO", "50"))

# Adaptive threshold params
ADAPT_BLOCK = int(os.environ.get("ADAPT_BLOCK", "31"))  # must be odd
ADAPT_C = int(os.environ.get("ADAPT_C", "2"))

# Edge morphology tuning
EDGE_DILATE = int(os.environ.get("EDGE_DILATE", "1"))
EDGE_CLOSE = int(os.environ.get("EDGE_CLOSE", "2"))

# =========================
# HTTP upload helper
# =========================
def _encode_frame_jpeg(frame_bgr) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return b""
    return buf.tobytes()

def _post_image_to_api(image_bytes: bytes) -> Optional[Dict[str, Any]]:
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
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        print(f"[SCAN] HTTPError {exc.code}: {detail[:350]}", flush=True)
        return None
    except urllib.error.URLError as exc:
        print(f"[SCAN] URLError: {exc}", flush=True)
        return None
    except Exception as exc:
        print(f"[SCAN] parse failed: {exc}", flush=True)
        return None

# =========================
# Geometry helpers
# =========================
def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def polygon_area(pts: np.ndarray) -> float:
    pts = pts.reshape(-1, 2)
    x = pts[:, 0]; y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def compute_angle_deg(quad: np.ndarray) -> float:
    tl, tr = quad[0], quad[1]
    dx = float(tr[0] - tl[0])
    dy = float(tr[1] - tl[1])
    return float(np.degrees(np.arctan2(dy, dx)))

def warp_board(frame_bgr: np.ndarray, quad: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    Hinv = cv2.getPerspectiveTransform(dst, quad.astype(np.float32))
    warped = cv2.warpPerspective(frame_bgr, H, (size, size))
    return warped, H, Hinv

def map_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)

# =========================
# Robust edge helpers
# =========================
def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lo, hi)

def hsv_blue_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([BLUE_H_LO, BLUE_S_LO, BLUE_V_LO], dtype=np.uint8)
    upper = np.array([BLUE_H_HI, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def adaptive_lines(gray: np.ndarray) -> np.ndarray:
    blk = ADAPT_BLOCK if ADAPT_BLOCK % 2 == 1 else ADAPT_BLOCK + 1
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blk,
        ADAPT_C
    )
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    return th

def edges_combo(frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    stages = {}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    stages["1_gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # blur nhẹ hơn để không mất nét
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    stages["2_blur"] = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

    # auto canny trên gray
    e_gray = auto_canny(blur, sigma=0.33)
    stages["3_edges_gray"] = cv2.cvtColor(e_gray, cv2.COLOR_GRAY2BGR)

    # adaptive threshold bắt nét bút / đường tối
    th = adaptive_lines(gray)
    stages["2b_adapt_thresh"] = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    # blue mask bắt đường xanh (grid)
    m_blue = hsv_blue_mask(frame_bgr)
    stages["2hsv_blue_mask"] = cv2.cvtColor(m_blue, cv2.COLOR_GRAY2BGR)

    # edges từ blue mask (bằng canny hoặc gradient)
    m_blur = cv2.GaussianBlur(m_blue, (5, 5), 0)
    e_blue = cv2.Canny(m_blur, 40, 120)
    stages["3b_edges_blue"] = cv2.cvtColor(e_blue, cv2.COLOR_GRAY2BGR)

    # OR tất cả lại
    combo = cv2.bitwise_or(e_gray, e_blue)
    combo = cv2.bitwise_or(combo, th)
    stages["3c_edges_combo"] = cv2.cvtColor(combo, cv2.COLOR_GRAY2BGR)

    # nối cạnh bị đứt
    k = np.ones((3, 3), np.uint8)
    if EDGE_DILATE > 0:
        combo = cv2.dilate(combo, k, iterations=EDGE_DILATE)
    if EDGE_CLOSE > 0:
        combo = cv2.morphologyEx(combo, cv2.MORPH_CLOSE, k, iterations=EDGE_CLOSE)
    stages["3d_edges_closed"] = cv2.cvtColor(combo, cv2.COLOR_GRAY2BGR)

    return combo, stages

def find_board_quad_from_edges(edges: np.ndarray) -> Optional[np.ndarray]:
    """
    Tìm contour lớn nhất và approx 4 điểm.
    Dùng edges đã được "closed" => ổn định hơn.
    """
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H, W = edges.shape[:2]
    img_area = float(H * W)

    best = None
    best_area = 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.03 * img_area:
            continue

        peri = cv2.arcLength(c, True)
        if peri < 400:
            continue

        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            # thử nới lỏng chút
            approx2 = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx2) != 4:
                continue
            approx = approx2

        pts = approx.reshape(4, 2).astype(np.float32)

        # aspect ratio check
        x, y, w, h = cv2.boundingRect(approx)
        if h <= 1 or w <= 1:
            continue
        ar = w / float(h)
        if not (0.55 <= ar <= 1.9):
            continue

        polyA = polygon_area(pts)
        if polyA > best_area:
            best_area = polyA
            best = pts

    if best is None:
        return None
    return order_points(best)

# =========================
# Vision helpers (warp stages)
# =========================
def enhance_for_server(warp_bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(warp_bgr, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

def mask_white_objects(warp_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 160], dtype=np.uint8)
    upper = np.array([180, 70, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def detect_O_centers(warp_bgr: np.ndarray, rows: int, cols: int) -> List[Tuple[int, int]]:
    mask = mask_white_objects(warp_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    size = warp_bgr.shape[0]
    cell = size / float(max(1, cols))
    min_r = MIN_R_RATIO * cell
    max_r = MAX_R_RATIO * cell

    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 80:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < min_r or r > max_r:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < 0.45:
            continue
        centers.append((int(round(x)), int(round(y))))
    return centers

def detect_X_cells_simple(warp_bgr: np.ndarray, rows: int, cols: int) -> List[Tuple[int, int]]:
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(blur, sigma=0.33)

    H, W = edges.shape[:2]
    cell_w = W / float(cols)
    cell_h = H / float(rows)

    hits = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(c * cell_w); x1 = int((c + 1) * cell_w)
            y0 = int(r * cell_h); y1 = int((r + 1) * cell_h)

            roi = edges[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            lines = cv2.HoughLinesP(
                roi, 1, np.pi / 180,
                threshold=22,
                minLineLength=int(0.35 * min(roi.shape[:2])),
                maxLineGap=12
            )
            if lines is None:
                continue

            ang1 = 0
            ang2 = 0
            for l in lines[:, 0]:
                xA, yA, xB, yB = l
                dx = xB - xA
                dy = yB - yA
                a = np.degrees(np.arctan2(dy, dx))
                a = (a + 180) % 180
                if 25 <= a <= 70:
                    ang1 += 1
                elif 110 <= a <= 155:
                    ang2 += 1

            if ang1 >= 1 and ang2 >= 1:
                hits.append((r, c))
    return hits

# =========================
# Pipeline
# =========================
class ScanPipeline:
    def __init__(self, rows: int, cols: int, warp_size: int):
        self.rows = int(rows)
        self.cols = int(cols)
        self.warp_size = int(warp_size)

    def run(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        stages: Dict[str, np.ndarray] = {}

        # Build robust edges
        edges, st = edges_combo(frame_bgr)
        stages.update(st)

        # find quad
        quad = find_board_quad_from_edges(cv2.cvtColor(stages["3d_edges_closed"], cv2.COLOR_BGR2GRAY))
        if quad is None:
            # show debug board stage
            dbg = frame_bgr.copy()
            cv2.putText(dbg, "board quad NOT found", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            stages["4_board"] = dbg
            return {"found": False, "stages": stages}

        angle = compute_angle_deg(quad)

        board_vis = frame_bgr.copy()
        cv2.polylines(board_vis, [quad.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.putText(board_vis, f"angle={angle:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        stages["4_board"] = board_vis

        warp_bgr, H, Hinv = warp_board(frame_bgr, quad, self.warp_size)
        stages["5_warp"] = warp_bgr.copy()

        # local previews
        mask = mask_white_objects(warp_bgr)
        stages["6_mask_white"] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        o_centers = detect_O_centers(warp_bgr, self.rows, self.cols)
        x_cells = detect_X_cells_simple(warp_bgr, self.rows, self.cols)

        pieces_vis = warp_bgr.copy()
        s = self.warp_size
        for c in range(1, self.cols):
            xx = int(round(c * s / self.cols))
            cv2.line(pieces_vis, (xx, 0), (xx, s - 1), (255, 255, 0), 1)
        for r in range(1, self.rows):
            yy = int(round(r * s / self.rows))
            cv2.line(pieces_vis, (0, yy), (s - 1, yy), (255, 255, 0), 1)

        for (x, y) in o_centers:
            cv2.circle(pieces_vis, (x, y), 12, (0, 255, 0), 2)
            cv2.circle(pieces_vis, (x, y), 2, (0, 255, 0), -1)
        for (r, c) in x_cells:
            cx = int((c + 0.5) * s / self.cols)
            cy = int((r + 0.5) * s / self.rows)
            cv2.putText(pieces_vis, "X", (cx - 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        stages["7_pieces_local"] = pieces_vis

        # Prepare for server confirm
        warp_send = enhance_for_server(warp_bgr)
        stages["8_send"] = warp_send.copy()

        return {
            "found": True,
            "quad": quad,
            "angle": angle,
            "warp_send": warp_send,
            "Hinv": Hinv,
            "stages": stages
        }

# =========================
# Draw overlay on real camera using perspective mapping
# =========================
def draw_grid_perspective(frame_bgr: np.ndarray, Hinv: np.ndarray, rows: int, cols: int, size: int):
    overlay = frame_bgr.copy()
    for c in range(cols + 1):
        x = c * (size - 1) / float(cols)
        pts_w = np.array([[x, 0], [x, size - 1]], dtype=np.float32)
        pts_i = map_points(Hinv, pts_w).astype(np.int32)
        cv2.line(overlay, tuple(pts_i[0]), tuple(pts_i[1]), (0, 255, 255), 2)

    for r in range(rows + 1):
        y = r * (size - 1) / float(rows)
        pts_w = np.array([[0, y], [size - 1, y]], dtype=np.float32)
        pts_i = map_points(Hinv, pts_w).astype(np.int32)
        cv2.line(overlay, tuple(pts_i[0]), tuple(pts_i[1]), (0, 255, 255), 2)

    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

def fill_cell_perspective(frame_bgr: np.ndarray, Hinv: np.ndarray, r: int, c: int, rows: int, cols: int, size: int):
    x0 = c * (size - 1) / float(cols)
    x1 = (c + 1) * (size - 1) / float(cols)
    y0 = r * (size - 1) / float(rows)
    y1 = (r + 1) * (size - 1) / float(rows)

    poly_w = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    poly_i = map_points(Hinv, poly_w).astype(np.int32)

    overlay = frame_bgr.copy()
    cv2.fillPoly(overlay, [poly_i], (0, 255, 255))
    cv2.addWeighted(overlay, HILITE_ALPHA, frame_bgr, 1.0 - HILITE_ALPHA, 0, frame_bgr)

def board_from_server_cells(rows: int, cols: int, cells: List[Dict[str, Any]]) -> List[List[int]]:
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        r = int(cell.get("row", cell.get("r", 0)))
        c = int(cell.get("col", cell.get("c", 0)))
        st = str(cell.get("state", "empty"))
        if 0 <= r < rows and 0 <= c < cols:
            if st in ("player_x", "x", "X"):
                board[r][c] = 1
            elif st in ("robot_line", "o", "O", "player_o"):
                board[r][c] = 2
    return board

# =========================
# State store
# =========================
class BoardState:
    def __init__(self, rows: int, cols: int):
        self._lock = threading.Lock()
        self.rows = int(rows)
        self.cols = int(cols)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def snapshot(self) -> List[List[int]]:
        with self._lock:
            return [row[:] for row in self._board]

    def set_board(self, board: List[List[int]]):
        with self._lock:
            self._board = [row[:] for row in board]

    def stats(self):
        with self._lock:
            empty = sum(1 for r in self._board for v in r if v == 0)
            player = sum(1 for r in self._board for v in r if v == 1)
            robot = sum(1 for r in self._board for v in r if v == 2)
        return {"empty": empty, "player": player, "robot": robot}

# =========================
# Web + Camera
# =========================
class CameraWeb:
    def __init__(self, board: BoardState):
        self.board = board
        self.app = Flask("scan_board")

        self._lock = threading.Lock()
        self._last = None
        self._raw = None
        self._scan_status = "idle"
        self._stop = threading.Event()
        self._thread = None
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._play_requested = threading.Event()

        self._angle = None
        self._cells = []
        self._stages: Dict[str, np.ndarray] = {}

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

        @self.app.get("/state.json")
        def state():
            st = self.board.stats()
            with self._lock:
                st["scan_status"] = self._scan_status
                st["angle"] = self._angle
                st["cells"] = self._cells
                st["stages"] = list(self._stages.keys())
                st["rows"] = GRID_ROWS
                st["cols"] = GRID_COLS
                st["rotate180"] = ROTATE_180
            st["board"] = self.board.snapshot()
            return jsonify(st)

        @self.app.get("/play")
        def play():
            self._play_requested.set()
            return jsonify({"ok": True})

        @self.app.get("/mjpeg")
        def mjpeg():
            return Response(self._mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.get("/stage/<name>.jpg")
        def stage(name: str):
            with self._lock:
                img = self._stages.get(name)
            if img is None:
                blank = np.zeros((1, 1, 3), dtype=np.uint8)
                buf = cv2.imencode(".jpg", blank)[1].tobytes()
                return Response(buf, mimetype="image/jpeg")
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                return Response(b"", mimetype="image/jpeg")
            return Response(buf.tobytes(), mimetype="image/jpeg")

    def _html(self) -> str:
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Board 4x6</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }
    .wrap { display:flex; gap:16px; padding:16px; align-items:flex-start; }
    .card { background:#111827; border:1px solid #223; border-radius:12px; padding:12px; min-width:340px; }
    .kv { margin:8px 0; }
    .k { color:#93c5fd; }
    .row { display:flex; gap:8px; align-items:center; margin-top:10px; }
    .btn { background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:6px 10px; border-radius:8px; cursor:pointer; }
    .video { border:1px solid #223; border-radius:8px; width:__CAM_W__px; height:__CAM_H__px; object-fit:cover; }
    .cells { font-size:11px; color:#cbd5f5; white-space:pre; max-height:220px; overflow:auto; }

    .stages {
      width: 380px;
      background:#111827; border:1px solid #223; border-radius:12px; padding:12px;
    }
    .stageTitle { color:#93c5fd; margin:0 0 10px 0; font-weight:600; }
    .grid {
      display:grid;
      grid-template-columns: 1fr;
      gap:10px;
      max-height: calc(100vh - 80px);
      overflow:auto;
    }
    .thumb {
      border:1px solid #223;
      border-radius:10px;
      padding:8px;
      background:#0f1624;
    }
    .thumb h4 { margin:0 0 6px 0; font-size:12px; color:#cbd5f5; }
    .thumb img { width:100%; height:auto; border-radius:8px; display:block; }
    .note { font-size:12px; color:#aab7cf; margin-top:8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div>
      <img class="video" id="cam" src="/mjpeg" />
      <div class="note">rotate180: <span id="rot">?</span></div>
    </div>

    <div class="card">
      <div class="kv"><span class="k">Board:</span> <span id="gridrc">-</span></div>
      <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">idle</span></div>
      <div class="kv"><span class="k">Angle:</span> <span id="angle">-</span></div>

      <div class="kv"><span class="k">Empty:</span> <span id="empty">-</span></div>
      <div class="kv"><span class="k">Player X:</span> <span id="player">-</span></div>
      <div class="kv"><span class="k">Robot O:</span> <span id="robot">-</span></div>

      <div class="kv"><span class="k">Board state:</span></div>
      <div id="board" class="cells">-</div>

      <div class="kv"><span class="k">Cells (server):</span></div>
      <div id="cells" class="cells">-</div>

      <div class="row">
        <button class="btn" onclick="playScan()">Play Scan</button>
      </div>
    </div>

    <div class="stages">
      <div class="stageTitle">Processing Stages</div>
      <div id="stageGrid" class="grid"></div>
    </div>
  </div>

<script>
let stageNames = [];

function formatBoard(board) {
  if (!board || !board.length) return '-';
  return board.map(row => row.join(' ')).join('\\n');
}

function formatCells(cells) {
  if (!cells || !cells.length) return '-';
  return cells.map(c => `(${c.row ?? c.r},${c.col ?? c.c}) ${c.state}`).join('\\n');
}

function renderStages(names) {
  const grid = document.getElementById('stageGrid');
  grid.innerHTML = '';
  names.forEach(n => {
    const d = document.createElement('div');
    d.className = 'thumb';
    d.innerHTML = `<h4>${n}</h4><img src="/stage/${n}.jpg?ts=${Date.now()}" />`;
    grid.appendChild(d);
  });
}

async function tick() {
  try {
    const r = await fetch('/state.json', {cache:'no-store'});
    const js = await r.json();

    document.getElementById('gridrc').textContent = `rows=${js.rows}, cols=${js.cols}`;
    document.getElementById('scan_status').textContent = js.scan_status ?? '-';
    document.getElementById('angle').textContent = (js.angle ?? '-') + '';
    document.getElementById('rot').textContent = (js.rotate180 ? 'true' : 'false');

    document.getElementById('empty').textContent = js.empty ?? '-';
    document.getElementById('player').textContent = js.player ?? '-';
    document.getElementById('robot').textContent = js.robot ?? '-';

    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('cells').textContent = formatCells(js.cells || []);

    const names = js.stages || [];
    if (JSON.stringify(names) !== JSON.stringify(stageNames)) {
      stageNames = names;
      renderStages(stageNames);
    } else {
      const imgs = document.querySelectorAll('#stageGrid img');
      imgs.forEach(img => {
        const base = img.src.split('?')[0];
        img.src = base + '?ts=' + Date.now();
      });
    }
  } catch(e) {}
}

async function playScan() {
  try { await fetch('/play'); } catch(e) {}
  tick();
}

setInterval(tick, 450);
tick();
</script>
</body>
</html>
"""
        return html.replace("__CAM_W__", str(CAM_W)).replace("__CAM_H__", str(CAM_H))

    def _mjpeg_gen(self):
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._last is None else self._last.copy()

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
            time.sleep(0.03)

    def get_last_frame_raw(self):
        with self._lock:
            return None if self._raw is None else self._raw.copy()

    def set_live_frame(self, frame_bgr):
        with self._lock:
            self._last = frame_bgr

    def set_scan_status(self, status: str):
        with self._lock:
            self._scan_status = str(status)

    def set_pipeline_outputs(self, angle: Optional[float], cells: List[Dict[str, Any]], stages: Dict[str, np.ndarray]):
        with self._lock:
            self._angle = angle
            self._cells = cells
            self._stages = dict(sorted(stages.items(), key=lambda kv: kv[0]))

    def consume_scan_request(self) -> bool:
        if self._play_requested.is_set():
            self._play_requested.clear()
            return True
        return False

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
                time.sleep(0.05)
                continue

            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            with self._lock:
                self._raw = frame.copy()

            self.set_live_frame(frame)
            self._ready.set()
            time.sleep(0.01)

        cap.release()

    def start(self):
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

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
# Main
# =========================
def main():
    print("[START] scan_board_4x6_pipeline (robust edges)", flush=True)
    print(f"[CFG] GRID_ROWS={GRID_ROWS} GRID_COLS={GRID_COLS}", flush=True)
    print(f"[CFG] CAM={CAM_W}x{CAM_H}@{CAM_FPS} rotate180={ROTATE_180}", flush=True)
    print(f"[CFG] HSV blue H[{BLUE_H_LO},{BLUE_H_HI}] S>={BLUE_S_LO} V>={BLUE_V_LO}", flush=True)
    print(f"[CFG] ADAPT block={ADAPT_BLOCK} C={ADAPT_C}", flush=True)
    print(f"[CFG] EDGE dilate={EDGE_DILATE} close={EDGE_CLOSE}", flush=True)
    print(f"[CFG] SCAN_API_URL={SCAN_API_URL}", flush=True)

    board = BoardState(rows=GRID_ROWS, cols=GRID_COLS)
    cam = CameraWeb(board)
    cam.start()

    if not cam.wait_ready(timeout_sec=5.0):
        print("[CAM] not ready, stop", flush=True)
        return

    pipeline = ScanPipeline(rows=GRID_ROWS, cols=GRID_COLS, warp_size=WARP_SIZE)
    print("[WEB] press Play Scan to run pipeline", flush=True)

    try:
        while True:
            if cam.consume_scan_request():
                cam.set_scan_status("scanning")

                raw = cam.get_last_frame_raw()
                if raw is None:
                    cam.set_scan_status("failed")
                    time.sleep(0.2)
                    continue

                out = pipeline.run(raw)
                stages = out.get("stages", {})

                if not out.get("found"):
                    cam.set_pipeline_outputs(angle=None, cells=[], stages=stages)
                    cam.set_live_frame(stages.get("4_board", raw))
                    cam.set_scan_status("failed")
                    print("[SCAN] board quad not found", flush=True)
                    time.sleep(0.2)
                    continue

                angle = out["angle"]
                quad = out["quad"]
                Hinv = out["Hinv"]
                warp_send = out["warp_send"]

                cam.set_pipeline_outputs(angle=angle, cells=[], stages=stages)

                # send to server confirm
                img_bytes = _encode_frame_jpeg(warp_send)
                result = _post_image_to_api(img_bytes)

                live = raw.copy()
                cv2.polylines(live, [quad.astype(np.int32)], True, (0, 255, 255), 2)
                cv2.putText(live, f"angle={angle:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if not result or not result.get("found"):
                    cam.set_pipeline_outputs(angle=angle, cells=[], stages=stages)
                    cam.set_live_frame(live)
                    cam.set_scan_status("failed")
                    print("[SCAN] server confirm failed", flush=True)
                    time.sleep(0.2)
                    continue

                rows = int(result.get("rows", GRID_ROWS) or GRID_ROWS)
                cols = int(result.get("cols", GRID_COLS) or GRID_COLS)
                cells = result.get("cells", []) if isinstance(result.get("cells", []), list) else []

                board_mat = board_from_server_cells(rows, cols, cells)
                board.set_board(board_mat)

                # draw perspective grid and fill detected cells
                draw_grid_perspective(live, Hinv, rows, cols, WARP_SIZE)
                for r in range(rows):
                    for c in range(cols):
                        if board_mat[r][c] != 0:
                            fill_cell_perspective(live, Hinv, r, c, rows, cols, WARP_SIZE)

                cam.set_live_frame(live)
                cam.set_pipeline_outputs(angle=angle, cells=cells, stages=stages)
                cam.set_scan_status("ready")
                print("[SCAN] done", {"rows": rows, "cols": cols, "cells": len(cells), "angle": angle}, flush=True)

            time.sleep(0.08)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)

if __name__ == "__main__":
    main()
