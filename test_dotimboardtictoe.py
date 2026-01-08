#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import json
import uuid
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify

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

HTTP_TIMEOUT = float(os.environ.get("SCAN_TIMEOUT", "15"))


def _encode_frame_jpeg(frame_bgr) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return b""
    return buf.tobytes()


def _post_image_to_api(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    POST multipart/form-data:
      field name = "image"
    """
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
            "Accept": "application/json",
            "User-Agent": "MatthewPi/scan_board",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            status = getattr(resp, "status", 200)
            raw = resp.read().decode("utf-8", errors="ignore")
            if status < 200 or status >= 300:
                print(f"[SCAN] HTTP {status} body_head={raw[:200]!r}")
                return None
            try:
                return json.loads(raw)
            except Exception as exc:
                print(f"[SCAN] JSON parse failed: {exc} body_head={raw[:200]!r}")
                return None

    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            raw = ""
        print(f"[SCAN] HTTPError {exc.code}: {raw[:200]!r}")
        return None
    except urllib.error.URLError as exc:
        print(f"[SCAN] URLError: {exc}")
        return None
    except Exception as exc:
        print(f"[SCAN] request failed: {exc}")
        return None


def _bbox_to_pixels(bbox, w: int, h: int):
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox

    # normalized?
    if max(abs(float(x0)), abs(float(y0)), abs(float(x1)), abs(float(y1))) <= 1.5:
        x0 = int(round(float(x0) * w))
        x1 = int(round(float(x1) * w))
        y0 = int(round(float(y0) * h))
        y1 = int(round(float(y1) * h))
    else:
        x0 = int(round(float(x0)))
        x1 = int(round(float(x1)))
        y0 = int(round(float(y0)))
        y1 = int(round(float(y1)))

    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))

    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _cells_from_result(result, frame_shape):
    h, w = frame_shape[:2]
    cells = []
    for cell in result.get("cells", []) or []:
        bbox = _bbox_to_pixels(cell.get("bbox"), w, h)
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        state = str(cell.get("state", "empty")).strip().lower()
        # normalize state
        if state in ("x", "player", "playerx"):
            state = "player_x"
        if state in ("robot", "robotline", "line"):
            state = "robot_line"

        cells.append({
            "r": int(cell.get("row", 0)),
            "c": int(cell.get("col", 0)),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "state": state,
        })
    return cells


def _board_from_cells(rows: int, cols: int, cells):
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        r = cell.get("r", -1)
        c = cell.get("c", -1)
        st = cell.get("state", "empty")
        if 0 <= r < rows and 0 <= c < cols:
            if st == "player_x":
                board[r][c] = 1
            elif st == "robot_line":
                board[r][c] = 2
    return board


def draw_x_overlay_only(frame_bgr, cells=None):
    """
    ✅ chỉ vẽ quân X đã nhận dạng (player_x)
    """
    if not cells:
        return
    overlay = frame_bgr.copy()
    drew = 0
    for cell in cells:
        if cell.get("state") != "player_x":
            continue
        x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
        # chỉ highlight ô có X
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), -1)
        drew += 1

    if drew > 0:
        cv2.addWeighted(overlay, 0.30, frame_bgr, 0.70, 0, frame_bgr)


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


class CameraWeb:
    def __init__(self, board: BoardState):
        self.board = board
        self.app = Flask("scan_board")
        self._lock = threading.Lock()
        self._last = None
        self._cells = []
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
      <div class="kv"><span class="k">Player X:</span> <span id="player">-</span></div>
      <div class="kv"><span class="k">Robot:</span> <span id="robot">-</span></div>
      <div class="kv"><span class="k">Detected cells:</span> <span id="cells_count">-</span></div>
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
    document.getElementById('cells_count').textContent = cells.length ?? '-';
    document.getElementById('cells').textContent = formatCells(cells);
    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('scan_status').textContent = js.scan_status ?? '-';
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
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(0.03)

    def get_last_frame(self):
        with self._lock:
            return None if self._last is None else self._last.copy()

    def set_cells(self, cells):
        with self._lock:
            self._cells = cells

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

            with self._lock:
                cells = list(self._cells)

            # ✅ chỉ vẽ X
            if cells:
                draw_x_overlay_only(frame, cells=cells)

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


def main():
    print("[START] test_dotimboardtictoe", flush=True)
    board = BoardState()
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

                cells = _cells_from_result(result, frame.shape)
                board.set_board(_board_from_cells(rows, cols, cells))
                cam.set_cells(cells)

                cam.set_scan_status("ready")
                print(f"[SCAN] board updated: player_x={sum(1 for c in cells if c.get('state')=='player_x')}", flush=True)

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
