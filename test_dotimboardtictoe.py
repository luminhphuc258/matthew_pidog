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

# ✅ xoay 180
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

# Board 6x4
GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))

# warp size
CELL_PX = int(os.environ.get("CELL_PX", "90"))
WARP_W = GRID_COLS * CELL_PX
WARP_H = GRID_ROWS * CELL_PX

# endpoint
SCAN_API_URL = os.environ.get(
    "SCAN_API_URL",
    "https://embeddedprogramming-healtheworldserver.up.railway.app/scan_chess",
)

# smoothing quad
QUAD_SMOOTH_ALPHA = float(os.environ.get("QUAD_SMOOTH_ALPHA", "0.25"))

# yellow marker HSV (tune)
Y_H_MIN = int(os.environ.get("Y_H_MIN", "18"))
Y_H_MAX = int(os.environ.get("Y_H_MAX", "40"))
Y_S_MIN = int(os.environ.get("Y_S_MIN", "60"))
Y_S_MAX = int(os.environ.get("Y_S_MAX", "255"))
Y_V_MIN = int(os.environ.get("Y_V_MIN", "60"))
Y_V_MAX = int(os.environ.get("Y_V_MAX", "255"))

MARKER_MIN_AREA = int(os.environ.get("MARKER_MIN_AREA", "120"))  # giấy nhỏ -> hạ xuống
MARKER_MAX_AREA = int(os.environ.get("MARKER_MAX_AREA", "40000"))

# pad board ratio (bạn muốn 0)
BOARD_PAD_RATIO = float(os.environ.get("BOARD_PAD_RATIO", "0.0"))

# ✅ yêu cầu: nếu quad méo/không ra “chữ nhật đứng” => mở rộng ~ 1cm
# 1cm quy đổi pixel: tùy camera/độ gần, mặc định 12px; bạn có thể chỉnh
RECTIFY_PAD_PX = int(os.environ.get("RECTIFY_PAD_PX", "12"))

# refine theo edge (mở rộng dần tới khi "ăn" được nhiều edge biên trái/phải)
REFINE_STEPS = int(os.environ.get("REFINE_STEPS", "8"))
REFINE_STEP_PX = int(os.environ.get("REFINE_STEP_PX", "2"))

# web refresh state
STATE_POLL_MS = int(os.environ.get("STATE_POLL_MS", "500"))
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
# Geometry helpers
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


def poly_mask(shape_hw, quad: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    q = quad.astype(np.int32).reshape(4, 2)
    cv2.fillConvexPoly(m, q, 255)
    return m


def expand_quad_from_center(quad: np.ndarray, px: int) -> np.ndarray:
    """Expand quad outward from center by px (approx)."""
    cen = quad.mean(axis=0)
    v = quad - cen
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    v_unit = v / n
    return quad + v_unit * float(px)


def clip_quad(quad: np.ndarray, w: int, h: int) -> np.ndarray:
    q = quad.copy()
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    return q


# =========================
# Board detector
# =========================
class BoardDetector:
    def __init__(self, rows: int, cols: int):
        self.rows = int(rows)
        self.cols = int(cols)
        self.prev_quad: Optional[np.ndarray] = None  # TL TR BR BL

    def preprocess_edges(self, frame_bgr) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 60, 160)

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        return {"gray": gray, "blur": blur, "edges": edges, "edges_closed": closed}

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
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), ang = rect
            if w < 6 or h < 6:
                continue
            ar = max(w, h) / max(1.0, min(w, h))
            if ar > 8.0:
                continue

            box = cv2.boxPoints(rect).astype(np.float32)
            center = np.array([cx, cy], dtype=np.float32)
            cand.append({"area": float(area), "center": center, "box": box})

            cv2.drawContours(dbg, [box.astype(np.int32)], -1, (0, 0, 255), 2)
            cv2.circle(dbg, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        cand.sort(key=lambda x: x["area"], reverse=True)
        cand = cand[:4]

        return {"mask": mask, "candidates": cand, "debug": dbg}

    def _inner_corners_from_markers(self, candidates: List[Dict[str, Any]]):
        if not candidates:
            return None

        centers = np.array([c["center"] for c in candidates], dtype=np.float32)
        board_center = centers.mean(axis=0)

        inner_pts = []
        for c in candidates:
            box = c["box"]
            d = np.linalg.norm(box - board_center[None, :], axis=1)
            inner = box[int(np.argmin(d))]
            inner_pts.append(inner)

        inner_pts = np.array(inner_pts, dtype=np.float32)

        if len(inner_pts) == 4:
            return order_points_4(inner_pts)

        if len(inner_pts) == 3:
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
                slots[classify(p)] = p

            all_keys = {"tl", "tr", "br", "bl"}
            missing = list(all_keys - set(slots.keys()))
            if len(missing) != 1:
                return None

            miss = missing[0]
            tl = slots.get("tl")
            tr = slots.get("tr")
            br = slots.get("br")
            bl = slots.get("bl")

            inferred = None
            if miss == "tl" and tr is not None and br is not None and bl is not None:
                inferred = tr + bl - br
            elif miss == "tr" and tl is not None and br is not None and bl is not None:
                inferred = tl + br - bl
            elif miss == "br" and tl is not None and tr is not None and bl is not None:
                inferred = tr + bl - tl
            elif miss == "bl" and tl is not None and tr is not None and br is not None:
                inferred = tl + br - tr

            if inferred is None:
                p0, p1, p2 = inner_pts
                inferred = p0 + p2 - p1

            full = []
            for k in ["tl", "tr", "br", "bl"]:
                full.append(slots.get(k, inferred))
            return order_points_4(np.array(full, dtype=np.float32))

        return None

    def quad_from_markers(self, frame_bgr) -> Dict[str, Any]:
        mk = self.detect_yellow_markers(frame_bgr)
        quad = self._inner_corners_from_markers(mk["candidates"])

        src = "marker"
        if quad is None:
            if self.prev_quad is not None:
                quad = self.prev_quad.copy()
                src = "cache"
            else:
                return {"found": False, "quad": None, "src": "none",
                        "marker_mask": mk["mask"], "marker_dbg": mk["debug"]}

        if BOARD_PAD_RATIO > 0.0:
            cen = quad.mean(axis=0)
            quad = cen + (1.0 + BOARD_PAD_RATIO) * (quad - cen)

        if self.prev_quad is None:
            self.prev_quad = quad.copy()
        else:
            self.prev_quad = lerp_pts(self.prev_quad, quad, QUAD_SMOOTH_ALPHA)

        return {"found": True, "quad": self.prev_quad.copy(), "src": src,
                "marker_mask": mk["mask"], "marker_dbg": mk["debug"]}

    # -------------------------
    # RECTIFY: make "upright rectangle" in image plane
    # -------------------------
    def rectify_quad_to_upright_rect(self, frame_bgr, quad: np.ndarray, edges_closed: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Mục tiêu: biến vùng quad (từ marker) thành 1 hình chữ nhật đứng ổn định:
        - Fit minAreaRect dựa trên edge pixels trong vùng quad (cộng thêm padding px)
        - Sau đó refine trái/phải bằng cách mở rộng dần để bắt được nhiều edge biên.
        """
        h, w = frame_bgr.shape[:2]
        dbg = {}

        q0 = quad.copy()
        q0 = clip_quad(q0, w, h)

        # expand a bit so we include border edges
        q_pad = expand_quad_from_center(q0, RECTIFY_PAD_PX)
        q_pad = clip_quad(q_pad, w, h)

        mask = poly_mask((h, w), q_pad)
        edge_roi = cv2.bitwise_and(edges_closed, edges_closed, mask=mask)

        ys, xs = np.where(edge_roi > 0)
        if len(xs) < 200:
            # not enough edges => fallback to minAreaRect on quad points
            pts = q_pad.astype(np.float32)
            rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
            box = cv2.boxPoints(rect).astype(np.float32)
            rect_quad = order_points_4(box)
            dbg["rectify_mode"] = "minAreaRect_quad_fallback"
            return clip_quad(rect_quad, w, h), dbg

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
        box = cv2.boxPoints(rect).astype(np.float32)
        rect_quad = order_points_4(box)
        rect_quad = clip_quad(rect_quad, w, h)
        dbg["rectify_mode"] = "minAreaRect_edges"

        # refine: mở rộng trái/phải để “trúng” biên edge lớn
        def score_lr(q: np.ndarray) -> float:
            # lấy 2 dải sát biên trái và biên phải của rect và đếm edge
            m = poly_mask((h, w), q)
            e = cv2.bitwise_and(edges_closed, edges_closed, mask=m)

            # project to x: count edge pixels near left 8% and right 8%
            ys2, xs2 = np.where(e > 0)
            if len(xs2) < 50:
                return 0.0

            x_min = float(xs2.min())
            x_max = float(xs2.max())
            band = max(6.0, 0.08 * (x_max - x_min + 1.0))

            left_cnt = np.sum(xs2 <= x_min + band)
            right_cnt = np.sum(xs2 >= x_max - band)
            # ưu tiên có biên 2 bên
            return float(left_cnt + right_cnt)

        best = rect_quad.copy()
        best_score = score_lr(best)

        # mở rộng dần theo hướng normal từ center
        for _ in range(max(0, REFINE_STEPS)):
            cand = expand_quad_from_center(best, REFINE_STEP_PX)
            cand = clip_quad(cand, w, h)
            sc = score_lr(cand)
            if sc >= best_score * 1.02:  # chỉ nhận khi cải thiện rõ
                best = cand
                best_score = sc
            else:
                break

        dbg["rectify_best_score"] = best_score
        return best, dbg

    def warp_board(self, frame_bgr, quad: np.ndarray) -> np.ndarray:
        dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warp = cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))
        return warp

    def draw_quad(self, frame_bgr, quad: np.ndarray, text: str = "") -> np.ndarray:
        out = frame_bgr.copy()
        q = quad.astype(np.int32).reshape(4, 2)
        cv2.polylines(out, [q], True, (0, 255, 255), 2)
        for i, (x, y) in enumerate(q):
            cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(out, str(i), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if text:
            cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return out


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
        self.app = Flask("scan_board_6x4")
        self._lock = threading.Lock()

        self._last_frame = None
        self._scan_status = "idle"
        self._cells_server: List[Dict[str, Any]] = []

        # stages: only update after scan finished
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

        <div class="kv"><span class="k">Empty:</span> <span id="empty">-</span></div>
        <div class="kv"><span class="k">Player X:</span> <span id="player_x">-</span></div>
        <div class="kv"><span class="k">Robot O:</span> <span id="robot_o">-</span></div>

        <div class="kv"><span class="k">Board state:</span></div>
        <div id="board" class="mono">-</div>

        <div class="kv"><span class="k">Cells (server):</span></div>
        <div id="cells" class="mono" style="max-height:240px; overflow:auto;">-</div>

        <div class="kv">
          <button class="btn" onclick="playScan()">Play Scan</button>
          <div class="muted" style="margin-top:8px;">Stages chỉ update sau khi scan xong để đỡ giật.</div>
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
            jpgs[name] = _encode_jpeg(img_bgr, quality=80)
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
    print("[START] scan_board_6x4_send_boardquad_and_warp", flush=True)
    print(f"[CFG] SCAN_API_URL={SCAN_API_URL}", flush=True)
    print(f"[CFG] GRID_ROWS={GRID_ROWS} GRID_COLS={GRID_COLS}", flush=True)
    print(f"[CFG] ROTATE_180={ROTATE_180}", flush=True)
    print(f"[CFG] RECTIFY_PAD_PX={RECTIFY_PAD_PX} refine_steps={REFINE_STEPS} step_px={REFINE_STEP_PX}", flush=True)

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

            stages: Dict[str, np.ndarray] = {}

            # 1) edges
            pp = detector.preprocess_edges(frame)
            stages["3_edges_closed"] = pp["edges_closed"]  # vẫn show stage được
            stages["3m_marker_mask"] = None
            stages["3m_marker_debug"] = None

            # 2) quad from markers
            qinfo = detector.quad_from_markers(frame)
            stages["3m_marker_mask"] = qinfo.get("marker_mask")
            stages["3m_marker_debug"] = qinfo.get("marker_dbg")

            if not qinfo["found"] or qinfo["quad"] is None:
                dbg = frame.copy()
                cv2.putText(dbg, "board quad NOT found (markers)", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                stages["4_board_quad"] = dbg
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed")
                time.sleep(0.2)
                continue

            quad_raw = qinfo["quad"]

            # 3) rectify to upright rectangle-like (fit edges + expand a bit)
            quad_rect, dbg_info = detector.rectify_quad_to_upright_rect(frame, quad_raw, pp["edges_closed"])

            # 4) draw board quad (this is what you want yellow border to be)
            txt = f"src={qinfo['src']} pad={BOARD_PAD_RATIO} rectify={dbg_info.get('rectify_mode','?')}"
            board_quad_img = detector.draw_quad(frame, quad_rect, text=txt)
            stages["4_board_quad"] = board_quad_img

            # 5) warp using rectified quad
            warp = detector.warp_board(frame, quad_rect)
            stages["5_warp"] = warp

            # ✅ NEW: send ONLY 4_board_quad and 5_warp (no edges)
            # compose vertical to make server see both context + rectified view
            # resize board_quad_img to same width as warp for stable
            bw = warp.shape[1]
            bh = int(round(board_quad_img.shape[0] * (bw / max(1, board_quad_img.shape[1]))))
            board_small = cv2.resize(board_quad_img, (bw, bh))
            send_img = np.vstack([board_small, warp])
            stages["6_sent_image"] = send_img

            result = _post_image_to_api(_encode_jpeg(send_img, quality=80))

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

            print("[SCAN] ok", {"src": qinfo["src"], "cells": len(cells), "rectify": dbg_info}, flush=True)

            time.sleep(0.15)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
