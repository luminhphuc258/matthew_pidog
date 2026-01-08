cho mình full code mới từ code hiện tai, chỉnh theo ý của bạn sao cho nó dò tìm duoc tat cả quan x tren bàn cờ , chỉ vẽ quân x đã nhận dạng được còn lại ko vẽ gì hết #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

WEB_PORT = 8000
CAM_DEV = os.environ.get("CAM_DEV", "0")
CAM_W = int(os.environ.get("CAM_W", "320"))
CAM_H = int(os.environ.get("CAM_H", "240"))
CAM_FPS = int(os.environ.get("CAM_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

GRID_COLS = int(os.environ.get("GRID_COLS", "4"))
GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))
BLUE_LO = os.environ.get("BLUE_LO", "85,30,30")
BLUE_HI = os.environ.get("BLUE_HI", "140,255,255")
DIAG_MIN_RATIO = float(os.environ.get("DIAG_MIN_RATIO", "0.25"))
DIAG_SLOPE_MIN = float(os.environ.get("DIAG_SLOPE_MIN", "0.5"))
DIAG_SLOPE_MAX = float(os.environ.get("DIAG_SLOPE_MAX", "2.0"))


def _cluster_positions(pos, tol):
    if not pos:
        return []
    pos = sorted(pos)
    groups = [[pos[0]]]
    for p in pos[1:]:
        if abs(p - groups[-1][-1]) <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(sum(g) / len(g)) for g in groups]


def _pick_n(vals, n):
    if not vals:
        return []
    vals = sorted(vals)
    if len(vals) <= n:
        return vals
    if n == 1:
        return [vals[len(vals) // 2]]
    last = len(vals) - 1
    idxs = [int(round(i * last / (n - 1))) for i in range(n)]
    return [vals[i] for i in idxs]


def _evenly_spaced(min_v: int, max_v: int, count: int):
    if count <= 1:
        return [int(min_v)]
    if max_v <= min_v:
        return [int(min_v) for _ in range(count)]
    step = float(max_v - min_v) / float(count - 1)
    return [int(round(min_v + step * i)) for i in range(count)]


def _parse_hsv_triplet(val: str, fallback: Tuple[int, int, int]):
    try:
        parts = [int(x.strip()) for x in val.split(",")]
        if len(parts) == 3:
            return tuple(parts)
    except Exception:
        pass
    return fallback


def detect_blue_lines(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if h < 40 or w < 40:
        return []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = _parse_hsv_triplet(BLUE_LO, (85, 30, 30))
    hi = _parse_hsv_triplet(BLUE_HI, (140, 255, 255))
    blue_mask = cv2.inRange(hsv, lo, hi)
    blue_mask = cv2.medianBlur(blue_mask, 5)
    edges = cv2.Canny(blue_mask, 30, 120)
    min_len = int(min(w, h) * 0.18)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=min_len, maxLineGap=35)
    if lines is None:
        return []

    out = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = x2 - x1
        dy = y2 - y1
        if (dx * dx + dy * dy) ** 0.5 < min_len:
            continue
        out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


def build_grid_from_blue_lines(lines, frame_shape, cols: int = GRID_COLS, rows: int = GRID_ROWS):
    if hasattr(frame_shape, "shape"):
        h, w = frame_shape.shape[:2]
    else:
        h, w = frame_shape[:2]
    if not lines:
        return None
    v_pos = []
    h_pos = []
    for x1, y1, x2, y2 in lines:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < dy * 0.35:
            v_pos.append(int((x1 + x2) / 2))
        elif dy < dx * 0.35:
            h_pos.append(int((y1 + y2) / 2))
    tol = max(6, int(min(w, h) * 0.03))
    v_lines = _pick_n(_cluster_positions(v_pos, tol), cols + 1)
    h_lines = _pick_n(_cluster_positions(h_pos, tol), rows + 1)
    if len(v_lines) < 2 or len(h_lines) < 2:
        return None

    if len(v_lines) < cols + 1:
        vmin, vmax = min(v_lines), max(v_lines)
        v_lines = _evenly_spaced(vmin, vmax, cols + 1)
    if len(h_lines) < rows + 1:
        hmin, hmax = min(h_lines), max(h_lines)
        h_lines = _evenly_spaced(hmin, hmax, rows + 1)

    v_lines = sorted([max(0, min(w - 1, x)) for x in v_lines])
    h_lines = sorted([max(0, min(h - 1, y)) for y in h_lines])

    out_lines = []
    x0, x3 = v_lines[0], v_lines[-1]
    y0, y3 = h_lines[0], h_lines[-1]
    for x in v_lines:
        out_lines.append((x, y0, x, y3))
    for y in h_lines:
        out_lines.append((x0, y, x3, y))

    cells = []
    for r in range(rows):
        for c in range(cols):
            x_a, x_b = v_lines[c], v_lines[c + 1]
            y_a, y_b = h_lines[r], h_lines[r + 1]
            if x_b <= x_a or y_b <= y_a:
                continue
            cells.append({
                "r": r,
                "c": c,
                "x0": int(x_a),
                "y0": int(y_a),
                "x1": int(x_b),
                "y1": int(y_b),
                "cx": int((x_a + x_b) / 2),
                "cy": int((y_a + y_b) / 2),
            })

    return {"x": v_lines, "y": h_lines, "lines": out_lines, "cells": cells, "rows": rows, "cols": cols}


def detect_x_centers_contour(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if h < 40 or w < 40:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = _parse_hsv_triplet(BLUE_LO, (85, 30, 30))
    hi = _parse_hsv_triplet(BLUE_HI, (140, 255, 255))
    blue_mask = cv2.inRange(hsv, lo, hi)
    th[blue_mask > 0] = 0

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200 or area > 3000:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh) if bh else 0
        if 0.6 < ar < 1.6:
            centers.append((x + bw // 2, y + bh // 2))
    return centers


def detect_board_state(frame_bgr, grid, rows: int, cols: int):
    x_lines = grid["x"]
    y_lines = grid["y"]
    board = [[0 for _ in range(cols)] for _ in range(rows)]

    centers = detect_x_centers_contour(frame_bgr)
    for cx, cy in centers:
        c = -1
        r = -1
        for i in range(cols):
            if x_lines[i] <= cx < x_lines[i + 1]:
                c = i
                break
        for j in range(rows):
            if y_lines[j] <= cy < y_lines[j + 1]:
                r = j
                break
        if 0 <= r < rows and 0 <= c < cols:
            board[r][c] = 1
    return board


def draw_board_overlay(frame_bgr, blue_lines=None, cells=None, board=None):
    if board and cells:
        overlay = frame_bgr.copy()
        for cell in cells:
            r = cell["r"]
            c = cell["c"]
            if 0 <= r < len(board) and 0 <= c < len(board[r]) and board[r][c] == 1:
                x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0, frame_bgr)
    if blue_lines:
        for x1, y1, x2, y2 in blue_lines:
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (80, 220, 80), 1)
    if cells:
        for cell in cells:
            x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 255), 1)


class BoardState:
    def __init__(self, rows: int = GRID_ROWS, cols: int = GRID_COLS):
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
        return {"empty": empty, "player": player}


class CameraWeb:
    def __init__(self, board: BoardState):
        self.board = board
        self.app = Flask("scan_board")
        self._lock = threading.Lock()
        self._last = None
        self._grid = None
        self._cells = []
        self._p8_angle = 0
        self._p10_angle = 0
        self._stop = threading.Event()
        self._thread = None
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._play_requested = threading.Event()
        self._scan_status = "idle"

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

        @self.app.get("/state.json")
        def state():
            st = self.board.stats()
            st["p8_angle"] = self._p8_angle
            st["p10_angle"] = self._p10_angle
            with self._lock:
                st["grid_ok"] = self._grid is not None
                st["cells"] = self._cells
                st["scan_status"] = self._scan_status
            st["board"] = self.board.snapshot()
            return jsonify(st)

        @self.app.get("/play")
        def play():
            self._play_requested.set()
            return jsonify({"ok": True})

        @self.app.get("/p8")
        def p8():
            return jsonify({"ok": True, "p8_angle": self._p8_angle})

        @self.app.get("/p10")
        def p10():
            return jsonify({"ok": True, "p10_angle": self._p10_angle})

        @self.app.get("/mjpeg")
        def mjpeg():
            return Response(self._mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _html(self) -> str:
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Board</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }}
    .wrap {{ display:flex; gap:16px; padding:16px; align-items:flex-start; }}
    .card {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; min-width:260px; }}
    .kv {{ margin:8px 0; }}
    .k {{ color:#93c5fd; }}
    .row {{ display:flex; gap:8px; align-items:center; margin-top:10px; }}
    .btn {{ background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:6px 10px; border-radius:8px; cursor:pointer; }}
    .video {{ border:1px solid #223; border-radius:8px; width:{CAM_W}px; height:{CAM_H}px; }}
    .cells {{ font-size:11px; color:#cbd5f5; white-space:pre; max-height:260px; overflow:auto; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="kv"><span class="k">Empty cells:</span> <span id="empty">-</span></div>
      <div class="kv"><span class="k">Player moves:</span> <span id="player">-</span></div>
      <div class="kv"><span class="k">Detected cells:</span> <span id="cells_count">-</span></div>
      <div id="cells" class="cells">-</div>
      <div class="kv"><span class="k">Board state:</span></div>
      <div id="board" class="cells">-</div>
      <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">idle</span></div>
      <div class="row">
        <button class="btn" onclick="playScan()">Play</button>
      </div>
      <div class="kv"><span class="k">P8 angle:</span> <span id="p8">0</span></div>
      <div class="row">
        <button class="btn" onclick="p8Dec()">-</button>
        <button class="btn" onclick="p8Inc()">+</button>
      </div>
      <div class="kv" style="margin-top:12px;"><span class="k">P10 angle:</span> <span id="p10">0</span></div>
      <div class="row">
        <button class="btn" onclick="p10Dec()">-</button>
        <button class="btn" onclick="p10Inc()">+</button>
      </div>
    </div>
    <img class="video" id="cam" src="/mjpeg" />
  </div>
<script>
async function tick() {{
  try {{
    const r = await fetch('/state.json', {{cache:'no-store'}});
    const js = await r.json();
    document.getElementById('empty').textContent = js.empty ?? '-';
    document.getElementById('player').textContent = js.player ?? '-';
    const cells = js.cells || [];
    document.getElementById('cells_count').textContent = cells.length ?? '-';
    document.getElementById('cells').textContent = formatCells(cells);
    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('p8').textContent = js.p8_angle ?? '-';
    document.getElementById('p10').textContent = js.p10_angle ?? '-';
    document.getElementById('scan_status').textContent = js.scan_status ?? '-';
  }} catch(e) {{}}
}}
function formatCells(cells) {{
  if (!cells || !cells.length) return '-';
  return cells.map(c => `(${{c.r}},${{c.c}}) [${{c.x0}},${{c.y0}}]-[${{c.x1}},${{c.y1}}]`).join('\\n');
}}
function formatBoard(board) {{
  if (!board || !board.length) return '-';
  return board.map(row => row.join(' ')).join('\\n');
}}
async function playScan() {{
  try {{ await fetch('/play'); }} catch(e) {{}}
  tick();
}}
async function p8Inc() {{
  try {{ await fetch('/p8?action=inc'); }} catch(e) {{}}
  tick();
}}
async function p8Dec() {{
  try {{ await fetch('/p8?action=dec'); }} catch(e) {{}}
  tick();
}}
async function p10Inc() {{
  try {{ await fetch('/p10?action=inc'); }} catch(e) {{}}
  tick();
}}
async function p10Dec() {{
  try {{ await fetch('/p10?action=dec'); }} catch(e) {{}}
  tick();
}}
setInterval(tick, 500);
tick();
</script>
</body>
</html>"""

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

    def get_last_frame(self):
        with self._lock:
            return None if self._last is None else self._last.copy()

    def set_grid(self, grid):
        with self._lock:
            self._grid = grid
            self._cells = [] if not grid else grid.get("cells", [])

    def get_grid(self):
        with self._lock:
            return None if self._grid is None else dict(self._grid)

    def set_scan_status(self, status: str):
        with self._lock:
            self._scan_status = str(status)

    def consume_scan_request(self) -> bool:
        if self._play_requested.is_set():
            self._play_requested.clear()
            return True
        return False

    def _capture_loop(self):
        dev = int(CAM_DEV) if str(CAM_DEV).isdigit() else CAM_DEV
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[CAM] cannot open camera: {dev}")
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

            grid = self.get_grid()
            if grid:
                board = detect_board_state(frame, grid, self.board.rows, self.board.cols)
                self.board.set_board(board)
                draw_board_overlay(frame, blue_lines=grid.get("lines"), cells=grid.get("cells"), board=board)
            else:
                blue_lines = detect_blue_lines(frame)
                draw_board_overlay(frame, blue_lines=blue_lines, cells=None, board=None)

            with self._lock:
                self._last = frame
            self._ready.set()
            time.sleep(0.01)

        cap.release()

    def start(self):
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False, threaded=True),
            daemon=True,
        ).start()
        print(f"[WEB] http://<pi_ip>:{WEB_PORT}/", flush=True)

    def wait_ready(self, timeout_sec: float = 5.0) -> bool:
        self._ready.wait(timeout=max(0.1, float(timeout_sec)))
        return self._ready.is_set() and not self._failed.is_set()


def main():
    print("[START] test_sacnningboard", flush=True)
    board = BoardState()
    cam = CameraWeb(board)
    cam.start()
    if not cam.wait_ready(timeout_sec=5.0):
        print("[CAM] not ready, stop", flush=True)
        return

    print("[WEB] waiting for Play button to scan")
    try:
        while True:
            if cam.consume_scan_request():
                cam.set_scan_status("scanning")
                frame = cam.get_last_frame()
                if frame is not None:
                    lines = detect_blue_lines(frame)
                    grid = build_grid_from_blue_lines(lines, frame, cols=GRID_COLS, rows=GRID_ROWS)
                    if grid:
                        cam.set_grid(grid)
                        cam.set_scan_status("ready")
                        print("[SCAN] grid ready")
                    else:
                        cam.set_scan_status("failed")
                        print("[SCAN] grid not found")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
