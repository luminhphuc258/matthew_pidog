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
CAM_FPS = int(os.environ.get("CAM_FPS", "12"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))

# ✅ bạn đang muốn xoay 180 => default True
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

# Board 6x4 (rows=6, cols=4)
GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))

# output warp size (mỗi cell ~ px)
CELL_PX = int(os.environ.get("CELL_PX", "90"))
WARP_W = GRID_COLS * CELL_PX
WARP_H = GRID_ROWS * CELL_PX

# endpoint nhận ảnh composite để xác nhận lại state
SCAN_API_URL = os.environ.get(
    "SCAN_API_URL",
    "https://embeddedprogramming-healtheworldserver.up.railway.app/scan_chess",
)

# smoothing quad (0..1) - càng nhỏ càng “cố định”
QUAD_SMOOTH_ALPHA = float(os.environ.get("QUAD_SMOOTH_ALPHA", "0.25"))

# marker HSV for yellow sticky notes (tùy ánh sáng có thể chỉnh)
Y_H_MIN = int(os.environ.get("Y_H_MIN", "18"))
Y_H_MAX = int(os.environ.get("Y_H_MAX", "40"))
Y_S_MIN = int(os.environ.get("Y_S_MIN", "70"))
Y_S_MAX = int(os.environ.get("Y_S_MAX", "255"))
Y_V_MIN = int(os.environ.get("Y_V_MIN", "70"))
Y_V_MAX = int(os.environ.get("Y_V_MAX", "255"))

MARKER_MIN_AREA = int(os.environ.get("MARKER_MIN_AREA", "250"))   # nếu giấy nhỏ quá => giảm
MARKER_MAX_AREA = int(os.environ.get("MARKER_MAX_AREA", "40000"))

# Không mở rộng vùng marker/board (đúng yêu cầu)
BOARD_PAD_RATIO = float(os.environ.get("BOARD_PAD_RATIO", "0.0"))

# web refresh state
STATE_POLL_MS = int(os.environ.get("STATE_POLL_MS", "500"))

# ✅ giảm giật: mjpeg delay
MJPEG_SLEEP = float(os.environ.get("MJPEG_SLEEP", "0.06"))


# =========================
# HTTP upload helper
# =========================
def _encode_jpeg(img_bgr, quality=JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
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
# Geometry / utility
# =========================
def order_points_4(pts: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def lerp_pts(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def point_infer_missing_corner(tl, tr, br, bl, missing: str):
    # parallelogram inference
    if missing == "tl":
        return tr + bl - br
    if missing == "tr":
        return tl + br - bl
    if missing == "br":
        return tr + bl - tl
    if missing == "bl":
        return tl + br - tr
    return None


# =========================
# Board detector (marker-based)
# =========================
class BoardDetector:
    def __init__(self, rows: int, cols: int):
        self.rows = int(rows)
        self.cols = int(cols)
        self.prev_quad: Optional[np.ndarray] = None  # TL TR BR BL in image coords

    def preprocess_edges(self, frame_bgr) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 60, 160)

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # make thicker (for GPT / visualization)
        thick = cv2.dilate(closed, kernel, iterations=1)

        return {
            "gray": gray,
            "blur": blur,
            "edges": edges,
            "edges_closed": closed,
            "edges_thick": thick,
        }

    def detect_yellow_markers(self, frame_bgr) -> Dict[str, Any]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([Y_H_MIN, Y_S_MIN, Y_V_MIN], dtype=np.uint8)
        upper = np.array([Y_H_MAX, Y_S_MAX, Y_V_MAX], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cand = []
        dbg = frame_bgr.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area < MARKER_MIN_AREA or area > MARKER_MAX_AREA:
                continue
            rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
            (cx, cy), (w, h), ang = rect
            if w < 6 or h < 6:
                continue
            ar = max(w, h) / max(1.0, min(w, h))
            if ar > 6.0:
                continue

            box = cv2.boxPoints(rect).astype(np.float32)
            center = np.array([cx, cy], dtype=np.float32)
            cand.append({"area": float(area), "center": center, "box": box})

            # draw debug
            cv2.drawContours(dbg, [box.astype(np.int32)], -1, (0, 0, 255), 2)
            cv2.circle(dbg, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # keep top 4 by area (robust when there are small yellow noises)
        cand.sort(key=lambda x: x["area"], reverse=True)
        cand = cand[:4]

        return {"mask": mask, "candidates": cand, "debug": dbg}

    def _pick_inner_corners(self, candidates: List[Dict[str, Any]], img_w: int, img_h: int):
        """
        Mấu chốt: marker nhỏ ở góc.
        Ta lấy "đỉnh marker gần tâm board nhất" => inner corner.
        """
        if not candidates:
            return None, None

        centers = np.array([c["center"] for c in candidates], dtype=np.float32)
        board_center = centers.mean(axis=0)

        # for each marker: choose vertex closest to board_center
        inner_pts = []
        for c in candidates:
            box = c["box"]  # 4 points
            d = np.linalg.norm(box - board_center[None, :], axis=1)
            inner = box[int(np.argmin(d))]
            inner_pts.append(inner)

        inner_pts = np.array(inner_pts, dtype=np.float32)

        # if we have 4 => order
        if len(inner_pts) == 4:
            quad = order_points_4(inner_pts)
            return quad, board_center

        # if 3 => infer missing corner
        if len(inner_pts) == 3:
            # approximate which corner is missing by ordering with "virtual" 4th:
            # First, create TL/TR/BR/BL slots by nearest to ideal positions around centroid.
            cen = inner_pts.mean(axis=0)

            def classify(pt):
                x, y = pt
                if x < cen[0] and y < cen[1]:
                    return "tl"
                if x >= cen[0] and y < cen[1]:
                    return "tr"
                if x >= cen[0] and y >= cen[1]:
                    return "br"
                return "bl"

            slots = {}
            for p in inner_pts:
                key = classify(p)
                # if collision, keep the closer to that quadrant
                slots[key] = p

            all_keys = {"tl", "tr", "br", "bl"}
            missing_keys = list(all_keys - set(slots.keys()))
            if len(missing_keys) != 1:
                return None, board_center
            missing = missing_keys[0]

            # need all 3 corners to infer
            tl = slots.get("tl")
            tr = slots.get("tr")
            br = slots.get("br")
            bl = slots.get("bl")

            # infer using available combination
            inferred = None
            if missing == "tl" and tr is not None and br is not None and bl is not None:
                inferred = point_infer_missing_corner(None, tr, br, bl, "tl")
            elif missing == "tr" and tl is not None and br is not None and bl is not None:
                inferred = point_infer_missing_corner(tl, None, br, bl, "tr")
            elif missing == "br" and tl is not None and tr is not None and bl is not None:
                inferred = point_infer_missing_corner(tl, tr, None, bl, "br")
            elif missing == "bl" and tl is not None and tr is not None and br is not None:
                inferred = point_infer_missing_corner(tl, tr, br, None, "bl")

            if inferred is None:
                # fallback: parallelogram using any 3 points
                # choose missing = p0 + p2 - p1 with some permutation:
                p0, p1, p2 = inner_pts
                inferred = p0 + p2 - p1

            # build full 4 set
            full = []
            for k in ["tl", "tr", "br", "bl"]:
                if k in slots:
                    full.append(slots[k])
                else:
                    full.append(inferred)
            quad = order_points_4(np.array(full, dtype=np.float32))
            return quad, board_center

        return None, board_center

    def get_board_quad_from_markers(self, frame_bgr) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        mk = self.detect_yellow_markers(frame_bgr)
        quad, center = self._pick_inner_corners(mk["candidates"], w, h)

        quad_src = "marker"
        if quad is None:
            # fallback: keep previous (fixed region)
            if self.prev_quad is not None:
                quad = self.prev_quad.copy()
                quad_src = "cache"
            else:
                return {
                    "found": False,
                    "quad": None,
                    "src": "none",
                    "marker_dbg": mk["debug"],
                    "marker_mask": mk["mask"],
                }

        # optional pad (but you requested pad=0)
        if BOARD_PAD_RATIO > 0.0:
            cen = quad.mean(axis=0)
            quad = cen + (1.0 + BOARD_PAD_RATIO) * (quad - cen)

        # smooth / stabilize so it doesn't jump due to detection noise
        if self.prev_quad is None:
            self.prev_quad = quad.copy()
        else:
            self.prev_quad = lerp_pts(self.prev_quad, quad, QUAD_SMOOTH_ALPHA)

        return {
            "found": True,
            "quad": self.prev_quad.copy(),
            "src": quad_src,
            "marker_dbg": mk["debug"],
            "marker_mask": mk["mask"],
        }

    def warp_board(self, frame_bgr, quad: np.ndarray) -> np.ndarray:
        dst = np.array(
            [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warp = cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))
        return warp

    def draw_quad_on_frame(self, frame_bgr, quad: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
        out = frame_bgr.copy()
        q = quad.astype(np.int32).reshape(4, 2)
        cv2.polylines(out, [q], True, color, 2)
        for i, (x, y) in enumerate(q):
            cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(out, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out

    def make_cell_labels_overlay(self, warp_bgr: np.ndarray) -> np.ndarray:
        img = warp_bgr.copy()
        # grid lines
        for r in range(1, GRID_ROWS):
            y = int(round(r * (WARP_H / GRID_ROWS)))
            cv2.line(img, (0, y), (WARP_W - 1, y), (255, 255, 0), 1)
        for c in range(1, GRID_COLS):
            x = int(round(c * (WARP_W / GRID_COLS)))
            cv2.line(img, (x, 0), (x, WARP_H - 1), (255, 255, 0), 1)

        # labels
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x0 = int(round(c * (WARP_W / GRID_COLS)))
                y0 = int(round(r * (WARP_H / GRID_ROWS)))
                cv2.putText(img, f"({r},{c})", (x0 + 6, y0 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return img

    def make_composite_for_api(self, warp_bgr: np.ndarray, edges_thick: np.ndarray) -> np.ndarray:
        # edges_thick is 1-channel -> convert to BGR
        e = cv2.cvtColor(edges_thick, cv2.COLOR_GRAY2BGR)
        e = cv2.resize(e, (warp_bgr.shape[1], warp_bgr.shape[0]))

        left = self.make_cell_labels_overlay(warp_bgr)
        # make edges stronger (white lines)
        e2 = e.copy()
        # stack side-by-side
        comp = np.hstack([left, e2])
        return comp


# =========================
# Board State store
# =========================
class BoardState:
    def __init__(self, rows: int, cols: int):
        self._lock = threading.Lock()
        self.rows = int(rows)
        self.cols = int(cols)
        # 0 empty, 1 human X, 2 robot O
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.angle = None
        self.last_scan_ts = 0.0

    def snapshot(self) -> List[List[int]]:
        with self._lock:
            return [row[:] for row in self._board]

    def set_board(self, board: List[List[int]]):
        with self._lock:
            self._board = [row[:] for row in board]
            self.last_scan_ts = time.time()

    def set_angle(self, ang):
        with self._lock:
            self.angle = ang

    def stats(self):
        with self._lock:
            empty = sum(1 for r in self._board for v in r if v == 0)
            player_x = sum(1 for r in self._board for v in r if v == 1)
            robot_o = sum(1 for r in self._board for v in r if v == 2)
            ang = self.angle
            ts = self.last_scan_ts
        return {"empty": empty, "player_x": player_x, "robot_o": robot_o, "angle": ang, "last_scan_ts": ts}


def board_from_server_cells(rows: int, cols: int, cells: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Server states có thể khác nhau. Mình map linh hoạt:
      - 'player_x' => 1
      - 'robot_o' / 'robot_circle' / 'player_o' / 'o' => 2
      - còn lại => 0
    """
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        r = int(cell.get("row", cell.get("r", -1)))
        c = int(cell.get("col", cell.get("c", -1)))
        st = str(cell.get("state", "empty")).lower().strip()
        if 0 <= r < rows and 0 <= c < cols:
            if st in ("player_x", "x", "human_x"):
                board[r][c] = 1
            elif st in ("robot_o", "robot_circle", "player_o", "o", "circle"):
                board[r][c] = 2
            else:
                board[r][c] = 0
    return board


# =========================
# Web + Camera
# =========================
class CameraWeb:
    def __init__(self, board: BoardState):
        self.board = board
        self.app = Flask("scan_board_4x6")
        self._lock = threading.Lock()

        self._last_frame = None
        self._scan_status = "idle"

        # stages: only update AFTER scan finished
        self._stages_jpg: Dict[str, bytes] = {}
        self._cells_server: List[Dict[str, Any]] = []

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
                st["scan_status"] = self._scan_status
                st["rows"] = self.board.rows
                st["cols"] = self.board.cols
                st["cells_server"] = self._cells_server
                st["stages"] = list(self._stages_jpg.keys())
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
            with self._lock:
                data = self._stages_jpg.get(name)
            if not data:
                return ("not found", 404)
            return send_file(BytesIO(data), mimetype="image/jpeg")

    def _html(self) -> str:
        # IMPORTANT: dùng .format => escape { } bằng double {{ }}
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Board 6x4</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }}
    .wrap {{ display:grid; grid-template-columns: 1fr 420px; gap:14px; padding:14px; }}
    .left {{ display:flex; gap:14px; align-items:flex-start; }}
    .card {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; min-width:360px; }}
    .kv {{ margin:8px 0; }}
    .k {{ color:#93c5fd; }}
    .btn {{ background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:8px 12px; border-radius:10px; cursor:pointer; }}
    .video {{ border:1px solid #223; border-radius:10px; width:{cam_w}px; height:{cam_h}px; background:#000; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre; font-size:12px; color:#cbd5f5; }}
    .stages {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; max-height: calc(100vh - 28px); overflow:auto; }}
    .stageimg {{ width:100%; border-radius:10px; border:1px solid #223; margin-top:10px; }}
    .title {{ font-weight:700; margin-bottom:8px; }}
    .muted {{ color:#93a4b8; font-size:12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <div class="card">
        <div class="kv"><span class="k">Board:</span> rows=<span id="rows">-</span>, cols=<span id="cols">-</span></div>
        <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">-</span></div>
        <div class="kv"><span class="k">Angle:</span> <span id="angle">-</span></div>

        <div class="kv"><span class="k">Empty:</span> <span id="empty">-</span></div>
        <div class="kv"><span class="k">Player X:</span> <span id="player_x">-</span></div>
        <div class="kv"><span class="k">Robot O:</span> <span id="robot_o">-</span></div>

        <div class="kv"><span class="k">Board state:</span></div>
        <div id="board" class="mono">-</div>

        <div class="kv"><span class="k">Cells (server):</span></div>
        <div id="cells" class="mono" style="max-height:240px; overflow:auto;">-</div>

        <div class="kv">
          <button class="btn" onclick="playScan()">Play Scan</button>
          <div class="muted" style="margin-top:8px;">Tip: stages chỉ update sau khi scan xong để đỡ giật.</div>
        </div>
      </div>

      <img class="video" id="cam" src="/mjpeg" />
    </div>

    <div class="stages">
      <div class="title">Processing Stages</div>
      <div class="muted">Chỉ refresh ảnh stage khi scan xong.</div>
      <div id="stage_list"></div>
    </div>
  </div>

<script>
let last_ts = 0;

function formatBoard(board) {{
  if (!board || !board.length) return '-';
  return board.map(row => row.join('')).join('\\n');
}}

function formatCells(cells) {{
  if (!cells || !cells.length) return '-';
  return cells.map(c => `(${{c.row}},${{c.col}}) ${{c.state}}`).join('\\n');
}}

function renderStages(names) {{
  const host = document.getElementById('stage_list');
  host.innerHTML = '';
  (names || []).forEach(n => {{
    const t = document.createElement('div');
    t.className = 'muted';
    t.textContent = n;
    const img = document.createElement('img');
    img.className = 'stageimg';
    img.src = `/stage/${{n}}.jpg?ts=${{Date.now()}}`;
    host.appendChild(t);
    host.appendChild(img);
  }});
}}

async function tick() {{
  try {{
    const r = await fetch('/state.json', {{cache:'no-store'}});
    const js = await r.json();

    document.getElementById('rows').textContent = js.rows ?? '-';
    document.getElementById('cols').textContent = js.cols ?? '-';

    document.getElementById('scan_status').textContent = js.scan_status ?? '-';
    document.getElementById('angle').textContent = (js.angle==null ? '-' : js.angle.toFixed(2));

    document.getElementById('empty').textContent = js.empty ?? '-';
    document.getElementById('player_x').textContent = js.player_x ?? '-';
    document.getElementById('robot_o').textContent = js.robot_o ?? '-';

    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('cells').textContent = formatCells(js.cells_server);

    // chỉ reload stages khi scan timestamp thay đổi (scan xong)
    const ts = js.last_scan_ts || 0;
    if (ts > 0 && ts !== last_ts) {{
      last_ts = ts;
      renderStages(js.stages || []);
    }}
  }} catch(e) {{}}
}}

async function playScan() {{
  try {{ await fetch('/play'); }} catch(e) {{}}
  tick();
}}

setInterval(tick, {poll_ms});
tick();
</script>
</body>
</html>
"""
        return html.format(cam_w=CAM_W, cam_h=CAM_H, poll_ms=STATE_POLL_MS)

    def _mjpeg_gen(self):
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._last_frame is None else self._last_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            jpg = _encode_jpeg(frame, quality=JPEG_QUALITY)
            if not jpg:
                time.sleep(0.03)
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(MJPEG_SLEEP)

    def get_last_frame(self):
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def set_last_frame(self, frame):
        with self._lock:
            self._last_frame = frame

    def set_scan_status(self, status: str):
        with self._lock:
            self._scan_status = str(status)

    def set_stage_images(self, stages: Dict[str, np.ndarray]):
        """
        stages: name -> BGR or Gray
        store as jpg bytes
        """
        jpgs = {}
        for name, img in stages.items():
            if img is None:
                continue
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            jpgs[name] = _encode_jpeg(img_bgr, quality=80)
        with self._lock:
            self._stages_jpg = jpgs

    def set_cells_server(self, cells: List[Dict[str, Any]]):
        with self._lock:
            self._cells_server = cells or []

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

            self.set_last_frame(frame)
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

    def wait_ready(self, timeout_sec: float = 5.0) -> bool:
        self._ready.wait(timeout=max(0.1, float(timeout_sec)))
        return self._ready.is_set() and not self._failed.is_set()


# =========================
# MAIN LOOP
# =========================
def main():
    print("[START] scan_board_6x4_marker_fixed", flush=True)
    print(f"[CFG] SCAN_API_URL={SCAN_API_URL}", flush=True)
    print(f"[CFG] GRID_ROWS={GRID_ROWS} GRID_COLS={GRID_COLS}", flush=True)
    print(f"[CFG] ROTATE_180={ROTATE_180}", flush=True)
    print(f"[CFG] MARKER HSV H({Y_H_MIN}-{Y_H_MAX}) S({Y_S_MIN}-{Y_S_MAX}) V({Y_V_MIN}-{Y_V_MAX})", flush=True)
    print(f"[CFG] MARKER area min={MARKER_MIN_AREA} max={MARKER_MAX_AREA}", flush=True)
    print(f"[CFG] BOARD_PAD_RATIO={BOARD_PAD_RATIO} (you want 0.0)", flush=True)

    board = BoardState(rows=GRID_ROWS, cols=GRID_COLS)
    cam = CameraWeb(board)
    cam.start()

    if not cam.wait_ready(timeout_sec=5.0):
        print("[CAM] not ready, stop", flush=True)
        return

    detector = BoardDetector(rows=GRID_ROWS, cols=GRID_COLS)

    print("[WEB] press Play Scan to run pipeline", flush=True)

    try:
        while True:
            if not cam.consume_scan_request():
                time.sleep(0.08)
                continue

            cam.set_scan_status("scanning")

            frame = cam.get_last_frame()
            if frame is None:
                cam.set_scan_status("failed")
                time.sleep(0.2)
                continue

            # 1) preprocess edges (for stage + for GPT)
            pp = detector.preprocess_edges(frame)

            # 2) detect board quad from markers (stable, not affected by X/O)
            qinfo = detector.get_board_quad_from_markers(frame)

            stages: Dict[str, np.ndarray] = {}
            stages["1_blur"] = pp["blur"]
            stages["2_edges"] = pp["edges"]
            stages["3_edges_closed"] = pp["edges_closed"]
            stages["3m_marker_mask"] = qinfo.get("marker_mask")
            stages["3m_marker_debug"] = qinfo.get("marker_dbg")

            if not qinfo["found"] or qinfo["quad"] is None:
                # show current board overlay fail
                dbg = frame.copy()
                cv2.putText(dbg, "board quad NOT found (markers)", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                stages["4_board_quad"] = dbg
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed")
                time.sleep(0.2)
                continue

            quad = qinfo["quad"]

            # 3) draw quad on original
            board_dbg = detector.draw_quad_on_frame(frame, quad, color=(0, 255, 255))
            cv2.putText(board_dbg, f"src={qinfo['src']} pad={BOARD_PAD_RATIO}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            stages["4_board_quad"] = board_dbg

            # 4) warp to canonical rectangle (upright)
            warp = detector.warp_board(frame, quad)
            stages["5_warp"] = warp

            # 5) build composite image to send endpoint (warp+edges_thick)
            edges_warp = detector.warp_board(cv2.cvtColor(pp["edges_thick"], cv2.COLOR_GRAY2BGR), quad)
            edges_warp_g = cv2.cvtColor(edges_warp, cv2.COLOR_BGR2GRAY)
            composite = detector.make_composite_for_api(warp, edges_warp_g)
            stages["6_cells_composite_sent"] = composite

            # send composite
            result = _post_image_to_api(_encode_jpeg(composite, quality=80))

            if not result or not result.get("found"):
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed")
                time.sleep(0.2)
                continue

            # server cells
            cells = result.get("cells", []) or []
            cam.set_cells_server(cells)

            # board map (robot = O)
            board_mat = board_from_server_cells(GRID_ROWS, GRID_COLS, cells)
            board.set_board(board_mat)

            # angle info (optional)
            angle = result.get("angle", None)
            try:
                board.set_angle(float(angle) if angle is not None else None)
            except Exception:
                board.set_angle(None)

            # publish stages only once scan finished
            cam.set_stage_images(stages)
            cam.set_scan_status("ready")

            print("[SCAN] ok", {
                "src": qinfo["src"],
                "cells": len(cells),
                "angle": board.stats()["angle"],
            }, flush=True)

            time.sleep(0.15)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
