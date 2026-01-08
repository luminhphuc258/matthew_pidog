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
from flask import Flask, Response, jsonify

# =========================
# CONFIG
# =========================
WEB_PORT = 8000
CAM_DEV = os.environ.get("CAM_DEV", "0")
CAM_W = int(os.environ.get("CAM_W", "320"))
CAM_H = int(os.environ.get("CAM_H", "240"))
CAM_FPS = int(os.environ.get("CAM_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

GRID_COLS = int(os.environ.get("GRID_COLS", "4"))
GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))

SCAN_API_URL = os.environ.get(
    "SCAN_API_URL",
    "https://embeddedprogramming-healtheworldserver.up.railway.app/scan_chess",
)

# ✅ NEW: endpoint đã fix row/col rồi => client KHÔNG offset nữa
# Nếu bạn muốn, vẫn có thể set = 0/0; code này không dùng.
CELL_ROW_OFFSET = int(os.environ.get("CELL_ROW_OFFSET", "0"))
CELL_COL_OFFSET = int(os.environ.get("CELL_COL_OFFSET", "0"))

# Nếu overlay bị lệch pixel do camera crop/rotate, chỉnh 2 biến này:
DRAW_X_OFF_X = int(os.environ.get("DRAW_X_OFF_X", "0"))
DRAW_X_OFF_Y = int(os.environ.get("DRAW_X_OFF_Y", "0"))

# inset để highlight không che đường kẻ
HILITE_INSET_RATIO = float(os.environ.get("HILITE_INSET_RATIO", "0.12"))

# Nếu muốn co grid_bbox một chút (tránh bbox ăn nhầm viền):
GRID_INSET_TOP = float(os.environ.get("GRID_INSET_TOP", "0.00"))      # ratio 0..0.2
GRID_INSET_BOTTOM = float(os.environ.get("GRID_INSET_BOTTOM", "0.00"))# ratio 0..0.2
GRID_INSET_LEFT = float(os.environ.get("GRID_INSET_LEFT", "0.00"))    # ratio 0..0.2
GRID_INSET_RIGHT = float(os.environ.get("GRID_INSET_RIGHT", "0.00"))  # ratio 0..0.2


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
def _bbox_to_pixels(bbox, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Accept bbox normalized [0..1] or pixel coords.
    Return (x0,y0,x1,y1) in pixels.
    """
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(x) for x in bbox]
    except Exception:
        return None

    # normalized [0..1]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0 = int(round(x0 * w))
        x1 = int(round(x1 * w))
        y0 = int(round(y0 * h))
        y1 = int(round(y1 * h))
    else:
        x0 = int(round(x0))
        x1 = int(round(x1))
        y0 = int(round(y0))
        y1 = int(round(y1))

    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _apply_grid_inset_px(grid_bbox_px: Tuple[int, int, int, int], rows: int, cols: int) -> Tuple[int, int, int, int]:
    """
    Optional: shrink bbox a bit by ratios.
    """
    x0, y0, x1, y1 = grid_bbox_px
    W = max(1, x1 - x0)
    H = max(1, y1 - y0)

    dxL = int(round(W * max(0.0, GRID_INSET_LEFT)))
    dxR = int(round(W * max(0.0, GRID_INSET_RIGHT)))
    dyT = int(round(H * max(0.0, GRID_INSET_TOP)))
    dyB = int(round(H * max(0.0, GRID_INSET_BOTTOM)))

    x0 = x0 + dxL
    x1 = x1 - dxR
    y0 = y0 + dyT
    y1 = y1 - dyB

    if x1 <= x0 + 2:
        x0, x1 = grid_bbox_px[0], grid_bbox_px[2]
    if y1 <= y0 + 2:
        y0, y1 = grid_bbox_px[1], grid_bbox_px[3]
    return (x0, y0, x1, y1)


def _cells_from_result(result, frame_shape, rows: int, cols: int):
    """
    Parse cells từ API.
    ✅ Endpoint mới đã trả row/col chuẩn => KHÔNG offset.
    """
    h, w = frame_shape[:2]
    cells = []

    for cell in result.get("cells", []):
        bbox_px = _bbox_to_pixels(cell.get("bbox"), w, h)
        if not bbox_px:
            continue

        # ✅ row/col từ server (đã recompute) => dùng thẳng
        r0 = int(cell.get("row", 0))
        c0 = int(cell.get("col", 0))

        # clamp cho chắc
        r0 = max(0, min(rows - 1, r0))
        c0 = max(0, min(cols - 1, c0))

        x0, y0, x1, y1 = bbox_px
        cells.append({
            "r": r0,
            "c": c0,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "state": str(cell.get("state", "empty")),
        })

    return cells


def _board_from_cells(rows: int, cols: int, cells):
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        r = cell.get("r", -1)
        c = cell.get("c", -1)
        state = cell.get("state", "empty")
        if 0 <= r < rows and 0 <= c < cols:
            if state == "player_x":
                board[r][c] = 1
            elif state == "robot_line":
                board[r][c] = 2
    return board


# =========================
# Drawing helpers
# =========================
def draw_grid_bbox(frame_bgr, grid_bbox_px, rows, cols):
    if not grid_bbox_px or rows <= 0 or cols <= 0:
        return

    x0, y0, x1, y1 = grid_bbox_px
    x0 += DRAW_X_OFF_X; x1 += DRAW_X_OFF_X
    y0 += DRAW_X_OFF_Y; y1 += DRAW_X_OFF_Y

    W = max(1, x1 - x0)
    H = max(1, y1 - y0)
    cell_w = W / float(cols)
    cell_h = H / float(rows)

    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 255), 1)
    for c in range(1, cols):
        xx = int(round(x0 + c * cell_w))
        cv2.line(frame_bgr, (xx, y0), (xx, y1), (0, 255, 255), 1)
    for r in range(1, rows):
        yy = int(round(y0 + r * cell_h))
        cv2.line(frame_bgr, (x0, yy), (x1, yy), (0, 255, 255), 1)


def draw_x_overlay_by_grid(frame_bgr, grid_bbox_px, rows, cols, board):
    if not grid_bbox_px or rows <= 0 or cols <= 0:
        return

    x0, y0, x1, y1 = grid_bbox_px
    x0 += DRAW_X_OFF_X; x1 += DRAW_X_OFF_X
    y0 += DRAW_X_OFF_Y; y1 += DRAW_X_OFF_Y

    W = max(1, x1 - x0)
    H = max(1, y1 - y0)
    cell_w = W / float(cols)
    cell_h = H / float(rows)

    overlay = frame_bgr.copy()
    drew = 0

    for r in range(rows):
        for c in range(cols):
            if board[r][c] != 1:
                continue

            cx0 = int(round(x0 + c * cell_w))
            cy0 = int(round(y0 + r * cell_h))
            cx1 = int(round(x0 + (c + 1) * cell_w))
            cy1 = int(round(y0 + (r + 1) * cell_h))

            inset_x = int(round((cx1 - cx0) * HILITE_INSET_RATIO))
            inset_y = int(round((cy1 - cy0) * HILITE_INSET_RATIO))
            cx0 += inset_x; cx1 -= inset_x
            cy0 += inset_y; cy1 -= inset_y

            cx0 = max(0, min(frame_bgr.shape[1] - 1, cx0))
            cx1 = max(0, min(frame_bgr.shape[1] - 1, cx1))
            cy0 = max(0, min(frame_bgr.shape[0] - 1, cy0))
            cy1 = max(0, min(frame_bgr.shape[0] - 1, cy1))
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), (0, 255, 255), -1)
            drew += 1

    if drew > 0:
        cv2.addWeighted(overlay, 0.28, frame_bgr, 0.72, 0, frame_bgr)


# =========================
# State store
# =========================
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

        self._cells = []
        self._grid_bbox_px = None
        self._grid_rows = GRID_ROWS
        self._grid_cols = GRID_COLS

        self._scan_status = "idle"
        self._stop = threading.Event()
        self._thread = None
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
                st["cells"] = self._cells
                st["scan_status"] = self._scan_status
                st["grid_bbox_px"] = self._grid_bbox_px
                st["rows"] = self._grid_rows
                st["cols"] = self._grid_cols
                st["draw_off"] = {"x": DRAW_X_OFF_X, "y": DRAW_X_OFF_Y}
                st["grid_inset"] = {
                    "top": GRID_INSET_TOP, "bottom": GRID_INSET_BOTTOM,
                    "left": GRID_INSET_LEFT, "right": GRID_INSET_RIGHT
                }
            st["board"] = self.board.snapshot()
            return jsonify(st)

        @self.app.get("/play")
        def play():
            self._play_requested.set()
            return jsonify({"ok": True})

        @self.app.get("/mjpeg")
        def mjpeg():
            return Response(self._mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _html(self) -> str:
        # dùng format để tránh lỗi f-string/JS template
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Board</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }}
    .wrap {{ display:flex; gap:16px; padding:16px; align-items:flex-start; }}
    .card {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; min-width:340px; }}
    .kv {{ margin:8px 0; }}
    .k {{ color:#93c5fd; }}
    .row {{ display:flex; gap:8px; align-items:center; margin-top:10px; }}
    .btn {{ background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:6px 10px; border-radius:8px; cursor:pointer; }}
    .video {{ border:1px solid #223; border-radius:8px; width:{cam_w}px; height:{cam_h}px; }}
    .cells {{ font-size:11px; color:#cbd5f5; white-space:pre; max-height:360px; overflow:auto; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="kv"><span class="k">Empty cells:</span> <span id="empty">-</span></div>
      <div class="kv"><span class="k">Player X:</span> <span id="player">-</span></div>
      <div class="kv"><span class="k">Robot:</span> <span id="robot">-</span></div>
      <div class="kv"><span class="k">Detected cells:</span> <span id="cells_count">-</span></div>
      <div class="kv"><span class="k">Grid:</span> <span id="gridrc">-</span></div>
      <div class="kv"><span class="k">Draw offset:</span> <span id="drawoff">-</span></div>
      <div class="kv"><span class="k">Grid inset:</span> <span id="inset">-</span></div>

      <div id="cells" class="cells">-</div>

      <div class="kv"><span class="k">Board state:</span></div>
      <div id="board" class="cells">-</div>

      <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">idle</span></div>
      <div class="row">
        <button class="btn" onclick="playScan()">Play</button>
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
    document.getElementById('robot').textContent = js.robot ?? '-';

    const cells = js.cells || [];
    document.getElementById('cells_count').textContent = cells.length || 0;
    document.getElementById('cells').textContent = formatCells(cells);

    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('scan_status').textContent = js.scan_status ?? '-';

    document.getElementById('gridrc').textContent = `rows=${{js.rows}}, cols=${{js.cols}}`;
    const d = js.draw_off || {{x:0,y:0}};
    document.getElementById('drawoff').textContent = `x=${{d.x}}, y=${{d.y}}`;
    const ins = js.grid_inset || {{}};
    document.getElementById('inset').textContent = `t=${{ins.top||0}} b=${{ins.bottom||0}} l=${{ins.left||0}} r=${{ins.right||0}}`;
  }} catch(e) {{}}
}}

function formatCells(cells) {{
  if (!cells || !cells.length) return '-';
  return cells.map(c => `(${{c.r}},${{c.c}}) ${{c.state}} [${{c.x0}},${{c.y0}}]-[${{c.x1}},${{c.y1}}]`).join('\\n');
}}

function formatBoard(board) {{
  if (!board || !board.length) return '-';
  return board.map(row => row.join(' ')).join('\\n');
}}

async function playScan() {{
  try {{ await fetch('/play'); }} catch(e) {{}}
  tick();
}}

setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""
        return html.format(cam_w=CAM_W, cam_h=CAM_H)

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

    def get_last_frame(self):
        with self._lock:
            return None if self._last is None else self._last.copy()

    def set_cells(self, cells):
        with self._lock:
            self._cells = cells

    def set_grid(self, grid_bbox_px, rows, cols):
        with self._lock:
            self._grid_bbox_px = grid_bbox_px
            self._grid_rows = int(rows)
            self._grid_cols = int(cols)

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

            # draw overlay based on current board + grid_bbox
            with self._lock:
                grid_bbox_px = self._grid_bbox_px
                rows = self._grid_rows
                cols = self._grid_cols

            board_snapshot = self.board.snapshot()

            if grid_bbox_px:
                draw_grid_bbox(frame, grid_bbox_px, rows, cols)
                draw_x_overlay_by_grid(frame, grid_bbox_px, rows, cols, board_snapshot)

            with self._lock:
                self._last = frame

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
# Main loop
# =========================
def main():
    print("[START] test_dotimboardticttoe", flush=True)
    print(f"[CFG] SCAN_API_URL={SCAN_API_URL}", flush=True)
    print(f"[CFG] GRID_ROWS={GRID_ROWS} GRID_COLS={GRID_COLS}", flush=True)
    print(f"[CFG] OFFSET row={CELL_ROW_OFFSET} col={CELL_COL_OFFSET} (NOTE: endpoint fixed, client does NOT apply offset)", flush=True)
    print(f"[CFG] DRAW_OFF x={DRAW_X_OFF_X} y={DRAW_X_OFF_Y}", flush=True)
    print(f"[CFG] GRID_INSET t={GRID_INSET_TOP} b={GRID_INSET_BOTTOM} l={GRID_INSET_LEFT} r={GRID_INSET_RIGHT}", flush=True)

    board = BoardState(rows=GRID_ROWS, cols=GRID_COLS)
    cam = CameraWeb(board)
    cam.start()

    if not cam.wait_ready(timeout_sec=5.0):
        print("[CAM] not ready, stop", flush=True)
        return

    print("[WEB] waiting for Play button to scan", flush=True)

    try:
        while True:
            if cam.consume_scan_request():
                cam.set_scan_status("scanning")

                frame = cam.get_last_frame()
                if frame is None:
                    cam.set_scan_status("failed")
                    time.sleep(0.2)
                    continue

                img_bytes = _encode_frame_jpeg(frame)
                result = _post_image_to_api(img_bytes)

                if not result or not result.get("found"):
                    cam.set_scan_status("failed")
                    print("[SCAN] no board found", flush=True)
                    time.sleep(0.2)
                    continue

                rows = int(result.get("rows", GRID_ROWS) or GRID_ROWS)
                cols = int(result.get("cols", GRID_COLS) or GRID_COLS)

                # board cells from API (row/col already fixed)
                cells = _cells_from_result(result, frame.shape, rows, cols)
                board_mat = _board_from_cells(rows, cols, cells)
                board.set_board(board_mat)
                cam.set_cells(cells)

                # grid bbox (normalized -> pixels)
                grid_bbox_px = _bbox_to_pixels(result.get("grid_bbox"), frame.shape[1], frame.shape[0])
                if grid_bbox_px:
                    grid_bbox_px = _apply_grid_inset_px(grid_bbox_px, rows, cols)
                cam.set_grid(grid_bbox_px, rows, cols)

                cam.set_scan_status("ready")
                print("[SCAN] board updated", {"rows": rows, "cols": cols, "cells": len(cells), "grid_bbox_px": grid_bbox_px}, flush=True)

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
