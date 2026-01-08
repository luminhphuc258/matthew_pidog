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

ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))

CELL_PX = int(os.environ.get("CELL_PX", "90"))
WARP_W = GRID_COLS * CELL_PX
WARP_H = GRID_ROWS * CELL_PX

SCAN_API_URL = os.environ.get(
    "SCAN_API_URL",
    "https://embeddedprogramming-healtheworldserver.up.railway.app/scan_chess",
)

STATE_POLL_MS = int(os.environ.get("STATE_POLL_MS", "500"))
MJPEG_SLEEP = float(os.environ.get("MJPEG_SLEEP", "0.06"))

# Yellow HSV
Y_H_MIN = int(os.environ.get("Y_H_MIN", "18"))
Y_H_MAX = int(os.environ.get("Y_H_MAX", "40"))
Y_S_MIN = int(os.environ.get("Y_S_MIN", "70"))
Y_S_MAX = int(os.environ.get("Y_S_MAX", "255"))
Y_V_MIN = int(os.environ.get("Y_V_MIN", "70"))
Y_V_MAX = int(os.environ.get("Y_V_MAX", "255"))

MARKER_MIN_AREA = int(os.environ.get("MARKER_MIN_AREA", "120"))
MARKER_MAX_AREA = int(os.environ.get("MARKER_MAX_AREA", "40000"))
MARKER_MAX_AR = float(os.environ.get("MARKER_MAX_AR", "6.0"))
MARKER_MIN_SIDE = int(os.environ.get("MARKER_MIN_SIDE", "8"))

# Reject noisy yellow too close to top edge
TOP_NOISE_REJECT_Y = float(os.environ.get("TOP_NOISE_REJECT_Y", "0.10"))
CORNER_BAND = float(os.environ.get("CORNER_BAND", "0.22"))

ALLOW_INFER_4TH = str(os.environ.get("ALLOW_INFER_4TH", "1")).lower() in ("1", "true", "yes", "on")

# smoothing (quad points) to reduce jitter
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", "0.30"))  # 0..1


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
# Geometry helpers
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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


def warp_from_quad(frame_bgr, quad4: np.ndarray) -> np.ndarray:
    quad4 = order_points_4(quad4)
    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad4.astype(np.float32), dst)
    return cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))


def smooth_pts(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return ((1 - alpha) * prev + alpha * cur).astype(np.float32)


# =========================
# Marker-only Detector (inner-corner quad)
# =========================
class MarkerBoardDetector:
    def __init__(self):
        self.prev_quad: Optional[np.ndarray] = None

    def _mask_yellow(self, frame_bgr) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([Y_H_MIN, Y_S_MIN, Y_V_MIN], dtype=np.uint8)
        upper = np.array([Y_H_MAX, Y_S_MAX, Y_V_MAX], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _extract_candidates(self, frame_bgr) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        mask = self._mask_yellow(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cand = []
        dbg = frame_bgr.copy()

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < MARKER_MIN_AREA or area > MARKER_MAX_AREA:
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            if rw < MARKER_MIN_SIDE or rh < MARKER_MIN_SIDE:
                continue

            ar = max(rw, rh) / max(1.0, min(rw, rh))
            if ar > MARKER_MAX_AR:
                continue

            # reject top noise
            if cy < TOP_NOISE_REJECT_Y * h:
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(dbg, [box], -1, (0, 0, 255), 2)
                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(dbg, "REJ_TOP", (int(cx) + 6, int(cy) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            box = cv2.boxPoints(rect).astype(np.float32)  # 4 corners
            cv2.drawContours(dbg, [box.astype(np.int32)], -1, (0, 0, 255), 2)
            cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            cand.append({
                "center": np.array([cx, cy], dtype=np.float32),
                "box": box,
                "area": area,
            })

        return {"mask": mask, "candidates": cand, "debug": dbg}

    def _pick_best_4_by_corners(self, candidates: List[Dict[str, Any]], w: int, h: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        corners = {
            "tl": np.array([0.0, 0.0], dtype=np.float32),
            "tr": np.array([float(w - 1), 0.0], dtype=np.float32),
            "br": np.array([float(w - 1), float(h - 1)], dtype=np.float32),
            "bl": np.array([0.0, float(h - 1)], dtype=np.float32),
        }

        band_x = CORNER_BAND * w
        band_y = CORNER_BAND * h

        def in_band(pt, key):
            x, y = float(pt[0]), float(pt[1])
            if key == "tl": return x <= band_x and y <= band_y
            if key == "tr": return x >= (w - band_x) and y <= band_y
            if key == "br": return x >= (w - band_x) and y >= (h - band_y)
            if key == "bl": return x <= band_x and y >= (h - band_y)
            return False

        picked = {}
        used = set()
        for key in ["tl", "tr", "br", "bl"]:
            best_i = None
            best_score = 1e18
            for i, c in enumerate(candidates):
                if i in used:
                    continue
                pt = c["center"]
                d = float(np.linalg.norm(pt - corners[key]))
                if in_band(pt, key):
                    d *= 0.65
                if d < best_score:
                    best_score = d
                    best_i = i
            if best_i is not None:
                picked[key] = candidates[best_i]
                used.add(best_i)

        out = []
        for key in ["tl", "tr", "br", "bl"]:
            if key in picked:
                out.append(picked[key])
        return out

    def _infer_4th_center(self, centers3: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(centers3) != 3:
            return None
        pts = np.array(centers3, dtype=np.float32)
        cen = pts.mean(axis=0)

        slots = {}
        for p in pts:
            if p[0] < cen[0] and p[1] < cen[1]:
                slots["tl"] = p
            elif p[0] >= cen[0] and p[1] < cen[1]:
                slots["tr"] = p
            elif p[0] >= cen[0] and p[1] >= cen[1]:
                slots["br"] = p
            else:
                slots["bl"] = p

        keys = {"tl", "tr", "br", "bl"}
        missing = list(keys - set(slots.keys()))
        if len(missing) != 1:
            p0, p1, p2 = pts
            return p0 + p2 - p1

        m = missing[0]
        tl = slots.get("tl"); tr = slots.get("tr"); br = slots.get("br"); bl = slots.get("bl")
        if m == "tl" and tr is not None and br is not None and bl is not None:
            return tr + bl - br
        if m == "tr" and tl is not None and br is not None and bl is not None:
            return tl + br - bl
        if m == "br" and tl is not None and tr is not None and bl is not None:
            return tr + bl - tl
        if m == "bl" and tl is not None and tr is not None and br is not None:
            return tl + br - tr
        return None

    def detect_board_quad(self, frame_bgr) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        ex = self._extract_candidates(frame_bgr)
        mask = ex["mask"]
        dbg = ex["debug"].copy()
        cand = ex["candidates"]

        # pick 4 markers
        picked = self._pick_best_4_by_corners(cand, w, h)
        if len(picked) < 4 and len(cand) >= 4:
            cand2 = sorted(cand, key=lambda x: x["area"], reverse=True)[:8]
            picked = self._pick_best_4_by_corners(cand2, w, h)

        # if 3 markers only -> infer 4th center (still better than nothing)
        inferred_center = None
        if len(picked) == 3 and ALLOW_INFER_4TH:
            inferred_center = self._infer_4th_center([c["center"] for c in picked])
            # we cannot infer box corners reliably, so we will fail (need 4 real markers)
            # return early with debug info
            return {
                "found": False,
                "mask": mask,
                "debug": dbg,
                "reason": "only_3_markers_found",
                "picked_centers": [c["center"] for c in picked],
                "inferred_center": inferred_center
            }

        if len(picked) < 4:
            return {"found": False, "mask": mask, "debug": dbg, "reason": "not_enough_markers"}

        # ----- stage: connect 4 centers polygon -----
        centers4 = np.array([c["center"] for c in picked], dtype=np.float32)
        centers4_ord = order_points_4(centers4)

        stage_centers_poly = frame_bgr.copy()
        for i, p in enumerate(centers4_ord):
            cv2.circle(stage_centers_poly, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            cv2.putText(stage_centers_poly, f"{i}", (int(p[0]) + 6, int(p[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.polylines(stage_centers_poly, [centers4_ord.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.putText(stage_centers_poly, "4_center_poly", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ----- compute inner corner of each marker -----
        # inner corner = corner of marker box closest to global center (mean of marker centers)
        global_c = centers4.mean(axis=0)

        inner_pts = []
        stage_inner = frame_bgr.copy()

        for idx, c in enumerate(picked):
            box = c["box"]  # 4 corners
            # choose corner closest to global center
            dists = np.linalg.norm(box - global_c.reshape(1, 2), axis=1)
            k = int(np.argmin(dists))
            inner = box[k]
            inner_pts.append(inner)

            cv2.drawContours(stage_inner, [box.astype(np.int32)], -1, (0, 0, 255), 2)
            cv2.circle(stage_inner, (int(inner[0]), int(inner[1])), 8, (255, 0, 255), -1)
            cv2.putText(stage_inner, f"in{idx}", (int(inner[0]) + 6, int(inner[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        inner4 = order_points_4(np.array(inner_pts, dtype=np.float32))

        # smooth quad to reduce jitter
        inner4 = smooth_pts(self.prev_quad, inner4, SMOOTH_ALPHA)
        self.prev_quad = inner4

        # ----- stage: board quad from inner corners -----
        stage_board_quad = frame_bgr.copy()
        cv2.polylines(stage_board_quad, [inner4.astype(np.int32)], True, (0, 255, 255), 3)
        for i, p in enumerate(inner4):
            cv2.circle(stage_board_quad, (int(p[0]), int(p[1])), 6, (255, 0, 255), -1)
            cv2.putText(stage_board_quad, f"Q{i}", (int(p[0]) + 6, int(p[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(stage_board_quad, "board_quad (inner corners)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return {
            "found": True,
            "mask": mask,
            "debug": dbg,
            "stage_centers_poly": stage_centers_poly,
            "stage_inner_debug": stage_inner,
            "stage_board_quad": stage_board_quad,
            "quad": inner4,
        }


# =========================
# Board State store
# =========================
class BoardState:
    def __init__(self, rows: int, cols: int):
        self._lock = threading.Lock()
        self.rows = int(rows)
        self.cols = int(cols)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.last_scan_ts = 0.0

    def snapshot(self) -> List[List[int]]:
        with self._lock:
            return [row[:] for row in self._board]

    def set_board(self, board: List[List[int]]):
        with self._lock:
            self._board = [row[:] for row in board]
            self.last_scan_ts = time.time()

    def stats(self):
        with self._lock:
            empty = sum(1 for r in self._board for v in r if v == 0)
            player_x = sum(1 for r in self._board for v in r if v == 1)
            robot_o = sum(1 for r in self._board for v in r if v == 2)
            ts = self.last_scan_ts
        return {"empty": empty, "player_x": player_x, "robot_o": robot_o, "last_scan_ts": ts}


def board_from_server_cells(rows: int, cols: int, cells: List[Dict[str, Any]]) -> List[List[int]]:
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
        self.app = Flask("scan_board_marker_inner_quad")
        self._lock = threading.Lock()

        self._last_frame = None
        self._scan_status = "idle"
        self._cells_server: List[Dict[str, Any]] = []
        self._stages_jpg: Dict[str, bytes] = {}

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
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Board 6x4 (marker inner-quad)</title>
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

        <div class="kv"><span class="k">Empty:</span> <span id="empty">-</span></div>
        <div class="kv"><span class="k">Player X:</span> <span id="player_x">-</span></div>
        <div class="kv"><span class="k">Robot O:</span> <span id="robot_o">-</span></div>

        <div class="kv"><span class="k">Board state:</span></div>
        <div id="board" class="mono">-</div>

        <div class="kv"><span class="k">Cells (server):</span></div>
        <div id="cells" class="mono" style="max-height:240px; overflow:auto;">-</div>

        <div class="kv">
          <button class="btn" onclick="playScan()">Play Scan</button>
          <div class="muted" style="margin-top:8px;">Stages chỉ update sau scan xong để đỡ giật.</div>
        </div>
      </div>

      <img class="video" id="cam" src="/mjpeg" />
    </div>

    <div class="stages">
      <div class="title">Processing Stages</div>
      <div class="muted">marker-only: dùng “inner corner” của marker để warp (giảm lệch row).</div>
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

    document.getElementById('empty').textContent = js.empty ?? '-';
    document.getElementById('player_x').textContent = js.player_x ?? '-';
    document.getElementById('robot_o').textContent = js.robot_o ?? '-';

    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('cells').textContent = formatCells(js.cells_server);

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

    def set_cells_server(self, cells: List[Dict[str, Any]]):
        with self._lock:
            self._cells_server = cells or []

    def set_stage_images(self, stages: Dict[str, np.ndarray]):
        jpgs = {}
        for name, img in stages.items():
            if img is None:
                continue
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            jpgs[name] = _encode_jpeg(img_bgr, quality=85)
        with self._lock:
            self._stages_jpg = jpgs

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
# MAIN
# =========================
def main():
    print("[START] marker_inner_quad + show_center_poly + show_final_sent", flush=True)
    print(f"[CFG] GRID_ROWS={GRID_ROWS} GRID_COLS={GRID_COLS}", flush=True)
    print(f"[CFG] ROTATE_180={ROTATE_180}", flush=True)

    board = BoardState(rows=GRID_ROWS, cols=GRID_COLS)
    cam = CameraWeb(board)
    cam.start()

    if not cam.wait_ready(timeout_sec=5.0):
        print("[CAM] not ready, stop", flush=True)
        return

    detector = MarkerBoardDetector()

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

            stages: Dict[str, np.ndarray] = {}

            det = detector.detect_board_quad(frame)
            stages["3m_marker_mask"] = det.get("mask")
            stages["3m_marker_debug"] = det.get("debug")

            if not det.get("found"):
                fail = frame.copy()
                cv2.putText(fail, f"board_quad NOT found ({det.get('reason','fail')})", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                stages["4_center_poly"] = fail
                stages["4b_inner_corner_debug"] = fail
                stages["4c_board_quad"] = fail
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed")
                time.sleep(0.2)
                continue

            stages["4_center_poly"] = det.get("stage_centers_poly")
            stages["4b_inner_corner_debug"] = det.get("stage_inner_debug")
            stages["4c_board_quad"] = det.get("stage_board_quad")

            quad = det["quad"]
            warp = warp_from_quad(frame, quad)
            stages["5_warp"] = warp

            # ✅ Final image sent to server (show on dashboard)
            board_quad_img = stages["4c_board_quad"]
            bw = warp.shape[1]
            bh = int(round(board_quad_img.shape[0] * (bw / max(1, board_quad_img.shape[1]))))
            board_small = cv2.resize(board_quad_img, (bw, bh))
            send_img = np.vstack([board_small, warp])
            stages["6_final_sent_to_server"] = send_img

            result = _post_image_to_api(_encode_jpeg(send_img, quality=85))

            if not result or not result.get("found"):
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed")
                time.sleep(0.2)
                continue

            cells = result.get("cells", []) or []
            cam.set_cells_server(cells)

            board_mat = board_from_server_cells(GRID_ROWS, GRID_COLS, cells)
            board.set_board(board_mat)

            cam.set_stage_images(stages)
            cam.set_scan_status("ready")

            print("[SCAN] ok", {"cells": len(cells)}, flush=True)
            time.sleep(0.15)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
