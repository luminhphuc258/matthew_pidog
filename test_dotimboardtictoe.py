#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import json
import uuid
import threading
import urllib.request
import urllib.error
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file, abort

# =========================
# CONFIG
# =========================
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))

CAM_DEV = os.environ.get("CAM_DEV", "0")
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))

# camera rotate 180
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

# board size: 6 rows x 4 cols
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))
GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))

# Endpoint nhận ảnh composite để GPT xác nhận nước cờ
SCAN_API_URL = os.environ.get(
    "SCAN_API_URL",
    "https://embeddedprogramming-healtheworldserver.up.railway.app/scan_chess",
)

# Expand vùng board một chút (thay vì “2cm” tuyệt đối -> dùng ratio theo kích thước quad)
# 0.04 ~ 4% chiều rộng/cao (thường ra gần vài cm tùy camera/khung hình)
BOARD_EXPAND_RATIO = float(os.environ.get("BOARD_EXPAND_RATIO", "0.04"))

# Warp size: mỗi cell bao nhiêu px
WARP_CELL = int(os.environ.get("WARP_CELL", "90"))
WARP_W = GRID_COLS * WARP_CELL
WARP_H = GRID_ROWS * WARP_CELL

# Marker vàng HSV threshold (tùy ánh sáng bạn có thể chỉnh)
YELLOW_H_MIN = int(os.environ.get("YELLOW_H_MIN", "18"))
YELLOW_H_MAX = int(os.environ.get("YELLOW_H_MAX", "40"))
YELLOW_S_MIN = int(os.environ.get("YELLOW_S_MIN", "90"))
YELLOW_S_MAX = int(os.environ.get("YELLOW_S_MAX", "255"))
YELLOW_V_MIN = int(os.environ.get("YELLOW_V_MIN", "90"))
YELLOW_V_MAX = int(os.environ.get("YELLOW_V_MAX", "255"))

# lọc marker theo diện tích (px^2) – vì marker của bạn “nhỏ”
MARKER_MIN_AREA = int(os.environ.get("MARKER_MIN_AREA", "80"))
MARKER_MAX_AREA = int(os.environ.get("MARKER_MAX_AREA", "25000"))

# =========================
# HTTP upload helper
# =========================
def _encode_jpeg(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else b""

def _post_image_to_api(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    if not image_bytes:
        return None

    boundary = uuid.uuid4().hex
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="scan.jpg"\r\n'
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        print(f"[SCAN] HTTPError {exc.code}: {detail[:300]}", flush=True)
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
def _order_points(pts: np.ndarray) -> np.ndarray:
    """pts shape (4,2) -> order: TL, TR, BR, BL"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _expand_quad(quad: np.ndarray, ratio: float) -> np.ndarray:
    """Nới quad ra xung quanh theo ratio"""
    quad = quad.astype(np.float32).reshape(4, 2)
    c = quad.mean(axis=0)
    v = quad - c
    return (c + v * (1.0 + max(0.0, ratio))).astype(np.float32)

def _poly_angle_deg(quad_ordered: np.ndarray) -> float:
    """góc của cạnh trên (TL->TR)"""
    tl, tr, br, bl = quad_ordered
    dx = float(tr[0] - tl[0])
    dy = float(tr[1] - tl[1])
    return math.degrees(math.atan2(dy, dx))

def _warp_perspective(img_bgr, quad_ordered, out_w, out_h):
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad_ordered.astype(np.float32), dst)
    warp = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
    return warp, M

# =========================
# Marker detection (yellow rectangles)
# =========================
def detect_yellow_markers(frame_bgr) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Return marker centers list, and debug mask image (BGR).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([YELLOW_H_MIN, YELLOW_S_MIN, YELLOW_V_MIN], dtype=np.uint8)
    upper = np.array([YELLOW_H_MAX, YELLOW_S_MAX, YELLOW_V_MAX], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: List[Tuple[int, int]] = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < MARKER_MIN_AREA or area > MARKER_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 2 or h <= 2:
            continue
        aspect = w / float(h)
        # marker là hình chữ nhật nhỏ -> cho phép range rộng
        if not (0.2 <= aspect <= 5.0):
            continue
        M = cv2.moments(c)
        if abs(M.get("m00", 0.0)) < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    # sort ổn định theo x
    centers.sort(key=lambda p: (p[0], p[1]))

    dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (cx, cy) in centers:
        cv2.circle(dbg, (cx, cy), 6, (0, 0, 255), 2)

    return centers, dbg

def infer_4th_from_3(points3: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Với 3 góc của hình chữ nhật:
    - lấy cặp xa nhất làm đường chéo (A,B)
    - điểm còn lại là C
    - suy ra D = A + B - C
    """
    p = [np.array([x, y], dtype=np.float32) for x, y in points3]
    d01 = np.linalg.norm(p[0] - p[1])
    d02 = np.linalg.norm(p[0] - p[2])
    d12 = np.linalg.norm(p[1] - p[2])
    # farthest pair
    if d01 >= d02 and d01 >= d12:
        A, B, C = p[0], p[1], p[2]
    elif d02 >= d01 and d02 >= d12:
        A, B, C = p[0], p[2], p[1]
    else:
        A, B, C = p[1], p[2], p[0]
    D = A + B - C
    return int(round(float(D[0]))), int(round(float(D[1])))

# =========================
# Image processing pipeline
# =========================
def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def make_stage_bgr(img_any) -> np.ndarray:
    if img_any is None:
        return None
    if len(img_any.shape) == 2:
        return cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR)
    return img_any

def stack_with_labels(images: List[Tuple[str, np.ndarray]], w: int = 520) -> np.ndarray:
    """
    Stack dọc các ảnh, resize về cùng width.
    """
    panels = []
    for name, img in images:
        if img is None:
            continue
        img = make_stage_bgr(img)
        h0, w0 = img.shape[:2]
        if w0 != w:
            scale = w / float(w0)
            img = cv2.resize(img, (w, int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(img, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(img)
    if not panels:
        return np.zeros((200, w, 3), dtype=np.uint8)
    return np.vstack(panels)

def compose_for_api(edges_closed_bgr, board_overlay_bgr, warp_bgr) -> np.ndarray:
    # ghép 3 ảnh thành 1 để gửi endpoint
    return stack_with_labels([
        ("3d_edges_closed", edges_closed_bgr),
        ("4_board", board_overlay_bgr),
        ("5_warp", warp_bgr),
    ], w=640)

# =========================
# Board state
# =========================
class BoardState:
    def __init__(self, rows: int, cols: int):
        self._lock = threading.Lock()
        self.rows = int(rows)
        self.cols = int(cols)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def snapshot(self):
        with self._lock:
            return [r[:] for r in self._board]

    def set_board(self, mat):
        with self._lock:
            self._board = [r[:] for r in mat]

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
        self.app = Flask("scan_board_4x6")

        self._lock = threading.Lock()
        self._last = None  # latest camera frame (BGR)
        self._scan_status = "idle"
        self._angle = None
        self._scan_id = 0
        self._cells = []  # from server response

        # stages: name -> jpeg bytes
        self._stages: Dict[str, bytes] = {}

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._play_requested = threading.Event()

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

        @self.app.get("/state.json")
        def state():
            st = self.board.stats()
            with self._lock:
                st.update({
                    "rows": self.board.rows,
                    "cols": self.board.cols,
                    "scan_status": self._scan_status,
                    "angle": self._angle,
                    "scan_id": self._scan_id,
                    "cells": self._cells,
                    "rotate180": ROTATE_180,
                    "stages": sorted(list(self._stages.keys())),
                })
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
        def stage(name):
            key = f"{name}"
            with self._lock:
                data = self._stages.get(key)
            if not data:
                abort(404)
            return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    def _html(self) -> str:
        # NOTE: dùng JS template string => phải tránh .format ăn mất {}
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Board 4x6</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }}
    .wrap {{ display:flex; gap:16px; padding:16px; align-items:flex-start; }}
    .left {{ flex: 1.4; }}
    .mid {{ flex: 0.9; min-width: 320px; }}
    .right {{ flex: 1.0; min-width: 360px; max-width: 520px; }}
    .card {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; }}
    .kv {{ margin:8px 0; }}
    .k {{ color:#93c5fd; }}
    .btn {{ background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:8px 12px; border-radius:10px; cursor:pointer; }}
    .video {{ width: 100%; max-width: 980px; border:1px solid #223; border-radius:12px; }}
    .cells {{ font-size:12px; color:#cbd5f5; white-space:pre; max-height:420px; overflow:auto; }}
    .stageBox {{ margin-top:10px; }}
    .stageImg {{ width:100%; border:1px solid #223; border-radius:12px; }}
    h3 {{ margin: 0 0 10px 0; color:#93c5fd; }}
    .small {{ font-size: 12px; opacity:0.85; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <img class="video" id="cam" src="/mjpeg" />
      <div class="small" id="rot"></div>
    </div>

    <div class="mid">
      <div class="card">
        <div class="kv"><span class="k">Board:</span> <span id="rc">-</span></div>
        <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">-</span></div>
        <div class="kv"><span class="k">Scan id:</span> <span id="scan_id">-</span></div>
        <div class="kv"><span class="k">Angle:</span> <span id="angle">-</span></div>
        <div class="kv"><span class="k">Empty:</span> <span id="empty">-</span></div>
        <div class="kv"><span class="k">Player X:</span> <span id="player">-</span></div>
        <div class="kv"><span class="k">Robot O:</span> <span id="robot">-</span></div>

        <div class="kv"><span class="k">Board state:</span></div>
        <div id="board" class="cells">-</div>

        <div class="kv"><span class="k">Cells (server):</span></div>
        <div id="cells" class="cells">-</div>

        <div style="margin-top:10px;">
          <button class="btn" onclick="playScan()">Play Scan</button>
        </div>
      </div>
    </div>

    <div class="right">
      <div class="card">
        <h3>Processing Stages</h3>
        <div id="stages"></div>
      </div>
    </div>
  </div>

<script>
function formatBoard(board) {{
  if (!board || !board.length) return '-';
  return board.map(row => row.join('')).join('\\n');
}}
function formatCells(cells) {{
  if (!cells || !cells.length) return '-';
  return cells.map(c => `(${{
    c.row ?? c.r
  }},${{
    c.col ?? c.c
  }}) ${{
    c.state ?? '-'
  }}`).join('\\n');
}}
async function playScan() {{
  try {{
    await fetch('/play', {{cache:'no-store'}});
  }} catch(e) {{}}
}}
function stageHtml(names, scanId) {{
  if (!names || !names.length) return '<div class="small">No stages yet</div>';
  return names.map(n => `
    <div class="stageBox">
      <div class="small">${{n}}</div>
      <img class="stageImg" src="/stage/${{n}}.jpg?sid=${{scanId}}&t=${{Date.now()}}" />
    </div>
  `).join('');
}}
async function tick() {{
  try {{
    const r = await fetch('/state.json', {{cache:'no-store'}});
    const js = await r.json();

    document.getElementById('rc').textContent = `rows=${{js.rows}}, cols=${{js.cols}}`;
    document.getElementById('scan_status').textContent = js.scan_status ?? '-';
    document.getElementById('scan_id').textContent = js.scan_id ?? '-';
    document.getElementById('angle').textContent = (js.angle === null || js.angle === undefined) ? '-' : js.angle;
    document.getElementById('empty').textContent = js.empty ?? '-';
    document.getElementById('player').textContent = js.player ?? '-';
    document.getElementById('robot').textContent = js.robot ?? '-';

    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('cells').textContent = formatCells(js.cells);

    document.getElementById('stages').innerHTML = stageHtml(js.stages, js.scan_id);
    document.getElementById('rot').textContent = `rotate180=${{js.rotate180}}`;
  }} catch(e) {{}}
}}
setInterval(tick, 600);
tick();
</script>
</body>
</html>"""

    def set_scan_status(self, status: str):
        with self._lock:
            self._scan_status = str(status)

    def set_angle(self, angle):
        with self._lock:
            self._angle = angle

    def bump_scan_id(self):
        with self._lock:
            self._scan_id += 1
            return self._scan_id

    def set_cells_server(self, cells):
        with self._lock:
            self._cells = cells or []

    def set_stage(self, name: str, img_bgr_or_gray):
        if img_bgr_or_gray is None:
            return
        bgr = make_stage_bgr(img_bgr_or_gray)
        data = _encode_jpeg(bgr)
        with self._lock:
            self._stages[name] = data

    def clear_stages(self):
        with self._lock:
            self._stages = {}

    def consume_scan_request(self) -> bool:
        if self._play_requested.is_set():
            self._play_requested.clear()
            return True
        return False

    def get_last_frame(self):
        with self._lock:
            return None if self._last is None else self._last.copy()

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
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)

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
                self._last = frame

            self._ready.set()
            time.sleep(0.01)

        cap.release()

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
        print("[WEB] press Play Scan to run pipeline", flush=True)

    def wait_ready(self, timeout_sec: float = 5.0) -> bool:
        self._ready.wait(timeout=max(0.1, float(timeout_sec)))
        return self._ready.is_set() and not self._failed.is_set()

# =========================
# Main scan logic
# =========================
def draw_quad(frame_bgr, quad_ordered, color=(0, 255, 255), thickness=2):
    q = quad_ordered.reshape(4, 2).astype(int)
    cv2.polylines(frame_bgr, [q], isClosed=True, color=color, thickness=thickness)

def clamp_pt(pt, w, h):
    x, y = int(pt[0]), int(pt[1])
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return (x, y)

def scan_pipeline(frame_bgr, cam: CameraWeb) -> Dict[str, Any]:
    """
    Output:
      - found (bool)
      - quad (np.array shape 4x2 ordered TL TR BR BL)
      - angle (float)
      - edges_closed, board_overlay, warp
      - server_result (dict or None)
    """
    H, W = frame_bgr.shape[:2]
    cam.clear_stages()

    # 1) gray
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cam.set_stage("1_gray", gray)

    # 2) blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    cam.set_stage("2_blur", blur)

    # 3) edges (auto canny)
    edges = auto_canny(blur, sigma=0.33)
    cam.set_stage("3_edges", edges)

    # 3c) combine edges + threshold (giúp nét rõ hơn)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3
    )
    combo = cv2.bitwise_or(edges, thr)
    cam.set_stage("3c_edges_combo", combo)

    # 3d) close/open để nối nét đường viền
    k = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(combo, cv2.MORPH_CLOSE, k, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cam.set_stage("3d_edges_closed", closed)

    # 3m) detect markers yellow
    marker_centers, marker_dbg = detect_yellow_markers(frame_bgr)
    cam.set_stage("3m_marker_mask", marker_dbg)

    quad = None
    quad_src = None

    # ---- Prefer marker-based quad ----
    if len(marker_centers) >= 4:
        pts = np.array(marker_centers[:4], dtype=np.float32)  # lấy 4 cái đầu (thường đủ)
        quad = _order_points(pts)
        quad_src = "markers4"
    elif len(marker_centers) == 3:
        p4 = infer_4th_from_3(marker_centers)
        pts = np.array(marker_centers + [p4], dtype=np.float32)
        quad = _order_points(pts)
        quad_src = "markers3->infer4"
    else:
        quad_src = f"markers={len(marker_centers)}"

    # ---- Fallback: find biggest quad from edges if marker thiếu ----
    if quad is None:
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best = approx
        if best is not None and best_area > 5000:
            pts = best.reshape(4, 2).astype(np.float32)
            quad = _order_points(pts)
            quad_src = "edges_quad"

    board_overlay = frame_bgr.copy()
    if quad is None:
        cv2.putText(board_overlay, "board quad NOT found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cam.set_stage("4_board", board_overlay)
        return {
            "found": False,
            "quad": None,
            "angle": None,
            "edges_closed": make_stage_bgr(closed),
            "board_overlay": board_overlay,
            "warp": None,
            "server_result": None,
            "quad_src": quad_src,
        }

    # expand quad a bit (your “2cm”)
    quad_exp = _expand_quad(quad, BOARD_EXPAND_RATIO)
    # clamp
    quad_exp = np.array([clamp_pt(p, W, H) for p in quad_exp], dtype=np.float32)
    quad_exp = _order_points(quad_exp)

    angle = _poly_angle_deg(quad_exp)

    draw_quad(board_overlay, quad_exp, (0, 255, 255), 2)
    # debug marker points
    for (cx, cy) in marker_centers:
        cv2.circle(board_overlay, (cx, cy), 6, (0, 0, 255), 2)

    cv2.putText(board_overlay, f"angle={angle:.1f} src={quad_src}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cam.set_stage("4_board", board_overlay)

    # warp
    warp, _M = _warp_perspective(frame_bgr, quad_exp, WARP_W, WARP_H)
    cam.set_stage("5_warp", warp)

    # composite send to endpoint
    edges_closed_bgr = make_stage_bgr(closed)
    composite = compose_for_api(edges_closed_bgr, board_overlay, warp)
    cam.set_stage("6_send_composite", composite)

    server_result = _post_image_to_api(_encode_jpeg(composite))

    return {
        "found": True,
        "quad": quad_exp,
        "angle": angle,
        "edges_closed": edges_closed_bgr,
        "board_overlay": board_overlay,
        "warp": warp,
        "server_result": server_result,
        "quad_src": quad_src,
    }

def board_from_server_cells(rows: int, cols: int, result: Dict[str, Any]) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """
    Giữ tương thích với server hiện tại:
    - result["cells"] = [{row,col,state,...}]
    state: "player_x" => 1, "robot_o"/"robot_line" => 2, "empty" => 0
    """
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    cells_out = []

    for c in (result or {}).get("cells", []) or []:
        r = int(c.get("row", c.get("r", 0)))
        k = int(c.get("col", c.get("c", 0)))
        st = str(c.get("state", "empty"))
        r = max(0, min(rows - 1, r))
        k = max(0, min(cols - 1, k))

        if st == "player_x":
            board[r][k] = 1
        elif st in ("robot_o", "robot_line", "robot"):
            board[r][k] = 2
        else:
            board[r][k] = 0

        cells_out.append({"row": r, "col": k, "state": st})

    return board, cells_out

# =========================
# Main loop
# =========================
def main():
    print("[START] scan board 4x6", flush=True)
    print(f"[CFG] SCAN_API_URL={SCAN_API_URL}", flush=True)
    print(f"[CFG] GRID_ROWS={GRID_ROWS} GRID_COLS={GRID_COLS}", flush=True)
    print(f"[CFG] CAM {CAM_W}x{CAM_H} fps={CAM_FPS} rotate180={ROTATE_180}", flush=True)
    print(f"[CFG] BOARD_EXPAND_RATIO={BOARD_EXPAND_RATIO}", flush=True)
    print(f"[CFG] WARP size={WARP_W}x{WARP_H} (cell={WARP_CELL})", flush=True)
    print(f"[CFG] Marker HSV H[{YELLOW_H_MIN},{YELLOW_H_MAX}] S[{YELLOW_S_MIN},{YELLOW_S_MAX}] V[{YELLOW_V_MIN},{YELLOW_V_MAX}]", flush=True)
    print(f"[CFG] Marker area [{MARKER_MIN_AREA},{MARKER_MAX_AREA}]", flush=True)

    board = BoardState(rows=GRID_ROWS, cols=GRID_COLS)
    cam = CameraWeb(board)
    cam.start()

    if not cam.wait_ready(timeout_sec=6.0):
        print("[CAM] not ready, stop", flush=True)
        return

    while True:
        if cam.consume_scan_request():
            scan_id = cam.bump_scan_id()
            cam.set_scan_status(f"scanning #{scan_id}")
            cam.set_angle(None)
            cam.set_cells_server([])

            frame = cam.get_last_frame()
            if frame is None:
                cam.set_scan_status("failed (no frame)")
                time.sleep(0.2)
                continue

            # run pipeline
            cam.set_scan_status(f"scanning #{scan_id} (pipeline)")
            out = scan_pipeline(frame, cam)

            if not out["found"]:
                cam.set_scan_status(f"failed #{scan_id} (board not found)")
                time.sleep(0.2)
                continue

            cam.set_angle(out["angle"])

            # apply server result to board
            srv = out.get("server_result")
            if not srv or not srv.get("found", True):
                # nếu server không trả 'found' thì coi như vẫn ok, nhưng cells rỗng
                cam.set_scan_status(f"ready #{scan_id} (no server cells)")
                time.sleep(0.2)
                continue

            bmat, cells = board_from_server_cells(GRID_ROWS, GRID_COLS, srv)
            board.set_board(bmat)
            cam.set_cells_server(cells)

            cam.set_scan_status(f"ready #{scan_id}")
            print("[SCAN] done", {
                "scan_id": scan_id,
                "angle": out["angle"],
                "quad_src": out["quad_src"],
                "markers": len(detect_yellow_markers(frame)[0]),
                "server_cells": len(cells),
            }, flush=True)

        time.sleep(0.05)

if __name__ == "__main__":
    main()
