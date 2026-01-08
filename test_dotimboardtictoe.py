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

# ✅ rotate 180 if needed
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

# Board is 6x4 (rows=6, cols=4)
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

# Web refresh
STATE_POLL_MS = int(os.environ.get("STATE_POLL_MS", "500"))
MJPEG_SLEEP = float(os.environ.get("MJPEG_SLEEP", "0.06"))

# ---------- Yellow marker HSV ----------
Y_H_MIN = int(os.environ.get("Y_H_MIN", "18"))
Y_H_MAX = int(os.environ.get("Y_H_MAX", "40"))
Y_S_MIN = int(os.environ.get("Y_S_MIN", "70"))
Y_S_MAX = int(os.environ.get("Y_S_MAX", "255"))
Y_V_MIN = int(os.environ.get("Y_V_MIN", "70"))
Y_V_MAX = int(os.environ.get("Y_V_MAX", "255"))

# marker size filter
MARKER_MIN_AREA = int(os.environ.get("MARKER_MIN_AREA", "140"))     # tăng nhẹ để bớt nhiễu nhỏ
MARKER_MAX_AREA = int(os.environ.get("MARKER_MAX_AREA", "40000"))
MARKER_MAX_AR = float(os.environ.get("MARKER_MAX_AR", "6.0"))       # aspect ratio max
MARKER_MIN_SIDE = int(os.environ.get("MARKER_MIN_SIDE", "8"))       # w/h min

# “corner preference”
CORNER_BAND = float(os.environ.get("CORNER_BAND", "0.22"))  # vùng góc (22% ảnh)
TOP_NOISE_REJECT_Y = float(os.environ.get("TOP_NOISE_REJECT_Y", "0.10"))  # loại nhiễu nằm quá sát mép trên (10%)

# board bbox padding (giữ 0 nếu muốn đúng “vùng góc”)
BBOX_PAD_PX = int(os.environ.get("BBOX_PAD_PX", "0"))  # bạn muốn fixed => 0

# if only 3 markers, infer the 4th using rectangle rule
ALLOW_INFER_4TH = str(os.environ.get("ALLOW_INFER_4TH", "1")).lower() in ("1", "true", "yes", "on")

# smooth bbox to reduce jitter
BBOX_SMOOTH_ALPHA = float(os.environ.get("BBOX_SMOOTH_ALPHA", "0.25"))

# NEW: max x mismatch between top and bottom on the same side (reject top noise)
TOP_X_MATCH_MAX = float(os.environ.get("TOP_X_MATCH_MAX", "0.18"))  # 18% ảnh (tune nếu cần)


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


def bbox_from_centers(centers: np.ndarray, w: int, h: int, pad_px: int = 0) -> Tuple[int, int, int, int]:
    xs = centers[:, 0]
    ys = centers[:, 1]
    x0 = int(np.floor(xs.min())) - pad_px
    x1 = int(np.ceil(xs.max())) + pad_px
    y0 = int(np.floor(ys.min())) - pad_px
    y1 = int(np.ceil(ys.max())) + pad_px
    x0 = clamp(x0, 0, w - 2)
    y0 = clamp(y0, 0, h - 2)
    x1 = clamp(x1, x0 + 2, w - 1)
    y1 = clamp(y1, y0 + 2, h - 1)
    return x0, y0, x1, y1


def warp_rect(frame_bgr, rect_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect_bbox
    src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))


def smooth_bbox(prev: Optional[Tuple[int, int, int, int]], cur: Tuple[int, int, int, int], alpha: float) -> Tuple[int, int, int, int]:
    if prev is None:
        return cur
    px0, py0, px1, py1 = prev
    cx0, cy0, cx1, cy1 = cur
    x0 = int(round((1 - alpha) * px0 + alpha * cx0))
    y0 = int(round((1 - alpha) * py0 + alpha * cy0))
    x1 = int(round((1 - alpha) * px1 + alpha * cx1))
    y1 = int(round((1 - alpha) * py1 + alpha * cy1))
    return (x0, y0, x1, y1)


# =========================
# Marker-only Detector
# =========================
class MarkerBoardDetector:
    def __init__(self):
        self.prev_bbox: Optional[Tuple[int, int, int, int]] = None

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

            # reject noisy yellow objects too close to top edge
            if cy < TOP_NOISE_REJECT_Y * h:
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(dbg, [box], -1, (0, 0, 255), 2)
                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(dbg, "REJ_TOP", (int(cx) + 6, int(cy) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue

            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(dbg, [box], -1, (0, 0, 255), 2)
            cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            cand.append({
                "center": np.array([cx, cy], dtype=np.float32),
                "area": area,
                "ar": float(ar),
            })

        return {"mask": mask, "candidates": cand, "debug": dbg}

    def _pick_best_4_by_bottom_then_topx(self, candidates: List[Dict[str, Any]], w: int, h: int) -> List[np.ndarray]:
        """
        FIX chính: chọn BL/BR trước, rồi chọn TL/TR theo tiêu chí:
        - TL: y nhỏ (top band), nằm nửa trái, và |x(TL)-x(BL)| nhỏ nhất
        - TR: y nhỏ, nằm nửa phải, và |x(TR)-x(BR)| nhỏ nhất
        => tránh lấy nhiễu ở mép trên (thường có x lệch xa so với bottom cùng bên)
        """
        if len(candidates) < 2:
            return []

        pts = [c["center"] for c in candidates]
        pts = np.array(pts, dtype=np.float32)

        top_y = CORNER_BAND * h
        bot_y = (1.0 - CORNER_BAND) * h
        mid_x = 0.5 * w

        # helper: pick nearest to target, with optional filters
        def pick_one(pool_idx, target_xy, extra_score_fn=None):
            best_i = None
            best_s = 1e18
            tx, ty = target_xy
            for i in pool_idx:
                x, y = float(pts[i][0]), float(pts[i][1])
                d = (x - tx) * (x - tx) + (y - ty) * (y - ty)
                s = d
                if extra_score_fn is not None:
                    s += float(extra_score_fn(x, y))
                if s < best_s:
                    best_s = s
                    best_i = i
            return best_i

        idx_all = list(range(len(pts)))

        # ----- pick BL, BR from bottom band -----
        idx_bottom = [i for i in idx_all if pts[i][1] >= bot_y]
        if len(idx_bottom) < 2:
            # fallback: just use all
            idx_bottom = idx_all[:]

        idx_left = [i for i in idx_bottom if pts[i][0] <= mid_x]
        idx_right = [i for i in idx_bottom if pts[i][0] >= mid_x]

        # BL: nearest (0,h)
        bl_i = pick_one(idx_left if idx_left else idx_bottom, (0.0, float(h - 1)))
        if bl_i is None:
            return []

        # BR: nearest (w,h), not same
        idx_right2 = [i for i in (idx_right if idx_right else idx_bottom) if i != bl_i]
        br_i = pick_one(idx_right2 if idx_right2 else [i for i in idx_bottom if i != bl_i], (float(w - 1), float(h - 1)))
        if br_i is None:
            return []

        bl = pts[bl_i]
        br = pts[br_i]

        # ----- pick TL, TR from top band, x-match with bottom -----
        idx_top = [i for i in idx_all if pts[i][1] <= top_y]
        if not idx_top:
            # fallback: allow a bit lower than band
            idx_top = [i for i in idx_all if pts[i][1] <= top_y * 1.25]

        idx_tl_pool = [i for i in idx_top if pts[i][0] <= mid_x and i not in (bl_i, br_i)]
        idx_tr_pool = [i for i in idx_top if pts[i][0] >= mid_x and i not in (bl_i, br_i)]

        max_dx = TOP_X_MATCH_MAX * w  # reject if top x lệch quá xa bottom x

        def tl_extra(x, y):
            # ưu tiên x gần BL nhất + y càng nhỏ càng tốt
            dx = abs(x - float(bl[0]))
            if dx > max_dx:
                return 1e9  # hard reject
            return (dx * dx) * 2.0 + (y * y) * 0.2

        def tr_extra(x, y):
            dx = abs(x - float(br[0]))
            if dx > max_dx:
                return 1e9
            return (dx * dx) * 2.0 + (y * y) * 0.2

        tl_i = pick_one(idx_tl_pool if idx_tl_pool else idx_top, (float(bl[0]), 0.0), extra_score_fn=tl_extra)
        tr_i = pick_one(idx_tr_pool if idx_tr_pool else idx_top, (float(br[0]), 0.0), extra_score_fn=tr_extra)

        # nếu trùng nhau hoặc None => fallback corner scoring cũ
        if tl_i is None or tr_i is None or tl_i == tr_i or tl_i in (bl_i, br_i) or tr_i in (bl_i, br_i):
            return []

        tl = pts[tl_i]
        tr = pts[tr_i]

        return [tl, tr, br, bl]

    def _pick_best_4_by_corners_fallback(self, candidates: List[Dict[str, Any]], w: int, h: int) -> List[np.ndarray]:
        """Fallback cũ (gần góc ảnh), có bonus corner band."""
        if not candidates:
            return []

        centers = [c["center"] for c in candidates]

        corners = {
            "tl": np.array([0.0, 0.0], dtype=np.float32),
            "tr": np.array([float(w - 1), 0.0], dtype=np.float32),
            "br": np.array([float(w - 1), float(h - 1)], dtype=np.float32),
            "bl": np.array([0.0, float(h - 1)], dtype=np.float32),
        }

        band_x = CORNER_BAND * w
        band_y = CORNER_BAND * h

        def in_corner_band(pt, key: str) -> bool:
            x, y = float(pt[0]), float(pt[1])
            if key == "tl":
                return x <= band_x and y <= band_y
            if key == "tr":
                return x >= (w - band_x) and y <= band_y
            if key == "br":
                return x >= (w - band_x) and y >= (h - band_y)
            if key == "bl":
                return x <= band_x and y >= (h - band_y)
            return False

        picked = {}
        used = set()

        for key in ["tl", "tr", "br", "bl"]:
            best_i = None
            best_score = 1e18
            for i, pt in enumerate(centers):
                if i in used:
                    continue
                d = float(np.linalg.norm(pt - corners[key]))
                if in_corner_band(pt, key):
                    d *= 0.65
                if key in ("tl", "bl") and pt[0] > w * 0.75:
                    d *= 2.0
                if key in ("tr", "br") and pt[0] < w * 0.25:
                    d *= 2.0
                if key in ("tl", "tr") and pt[1] > h * 0.75:
                    d *= 2.0
                if key in ("bl", "br") and pt[1] < h * 0.25:
                    d *= 2.0

                if d < best_score:
                    best_score = d
                    best_i = i

            if best_i is not None:
                picked[key] = centers[best_i]
                used.add(best_i)

        out = []
        for key in ["tl", "tr", "br", "bl"]:
            if key in picked:
                out.append(picked[key])
        return out

    def _infer_4th(self, pts3: List[np.ndarray]) -> Optional[np.ndarray]:
        """infer 4th point from 3 corners: p_missing = pA + pC - pB (parallelogram)"""
        if len(pts3) != 3:
            return None

        pts = np.array(pts3, dtype=np.float32)
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
        tl = slots.get("tl")
        tr = slots.get("tr")
        br = slots.get("br")
        bl = slots.get("bl")

        if m == "tl" and tr is not None and br is not None and bl is not None:
            return tr + bl - br
        if m == "tr" and tl is not None and br is not None and bl is not None:
            return tl + br - bl
        if m == "br" and tl is not None and tr is not None and bl is not None:
            return tr + bl - tl
        if m == "bl" and tl is not None and tr is not None and br is not None:
            return tl + br - tr

        return None

    def detect_board_bbox(self, frame_bgr) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]

        ex = self._extract_candidates(frame_bgr)
        cand = ex["candidates"]

        dbg = ex["debug"].copy()
        mask = ex["mask"]

        # ===== NEW pick logic (bottom first, then top-x match) =====
        picked = self._pick_best_4_by_bottom_then_topx(cand, w, h)

        # fallback to old if needed
        if len(picked) < 4:
            picked = self._pick_best_4_by_corners_fallback(cand, w, h)

        # if still not enough, try top by area subset then fallback
        if len(picked) < 4 and len(cand) >= 4:
            cand2 = sorted(cand, key=lambda x: x["area"], reverse=True)[:10]
            picked = self._pick_best_4_by_bottom_then_topx(cand2, w, h)
            if len(picked) < 4:
                picked = self._pick_best_4_by_corners_fallback(cand2, w, h)

        inferred = None
        if len(picked) == 3 and ALLOW_INFER_4TH:
            inferred = self._infer_4th(picked)
            if inferred is not None:
                picked = picked + [inferred]

        if len(picked) < 4:
            return {
                "found": False,
                "bbox": None,
                "mask": mask,
                "debug": dbg,
                "picked_centers": picked,
                "inferred": inferred
            }

        centers4 = np.array(picked[:4], dtype=np.float32)
        centers4 = order_points_4(centers4)

        # draw centers + label TL/TR/BR/BL for dễ nhìn
        labels = ["TL", "TR", "BR", "BL"]
        for i, p in enumerate(centers4):
            cv2.circle(dbg, (int(p[0]), int(p[1])), 7, (255, 0, 255), -1)
            cv2.putText(dbg, labels[i], (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        bbox = bbox_from_centers(centers4, w, h, pad_px=BBOX_PAD_PX)
        bbox = smooth_bbox(self.prev_bbox, bbox, BBOX_SMOOTH_ALPHA)
        self.prev_bbox = bbox

        x0, y0, x1, y1 = bbox

        out = frame_bgr.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(out, "board_bbox (marker-only)", (x0 + 10, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return {
            "found": True,
            "bbox": bbox,
            "mask": mask,
            "debug": dbg,
            "board_bbox_img": out,
            "centers4": centers4,
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
        self.app = Flask("scan_board_marker_only")
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
  <title>Scan Board 6x4 (marker-only)</title>
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
      <div class="muted">marker-only: không dùng edges. TL/TR chọn theo x gần BL/BR để tránh nhiễu top.</div>
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
    print("[START] marker_only_board_bbox_send_bbox_and_warp", flush=True)
    print(f"[CFG] GRID_ROWS={GRID_ROWS} GRID_COLS={GRID_COLS}", flush=True)
    print(f"[CFG] ROTATE_180={ROTATE_180}", flush=True)
    print(f"[CFG] HSV H[{Y_H_MIN},{Y_H_MAX}] S[{Y_S_MIN},{Y_S_MAX}] V[{Y_V_MIN},{Y_V_MAX}]", flush=True)
    print(f"[CFG] TOP_NOISE_REJECT_Y={TOP_NOISE_REJECT_Y} CORNER_BAND={CORNER_BAND}", flush=True)
    print(f"[CFG] TOP_X_MATCH_MAX={TOP_X_MATCH_MAX}", flush=True)

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

            det = detector.detect_board_bbox(frame)
            stages["3m_marker_mask"] = det.get("mask")
            stages["3m_marker_debug"] = det.get("debug")

            if not det["found"] or det.get("bbox") is None:
                fail = frame.copy()
                cv2.putText(fail, "board_bbox NOT found (markers)", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                stages["4_board_bbox"] = fail
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed")
                print("[SCAN] failed: no bbox", flush=True)
                time.sleep(0.2)
                continue

            bbox = det["bbox"]
            stages["4_board_bbox"] = det.get("board_bbox_img")

            warp = warp_rect(frame, bbox)
            stages["5_warp"] = warp

            # ✅ send ONLY 4_board_bbox and 5_warp (no edges)
            board_bbox_img = stages["4_board_bbox"]
            bw = warp.shape[1]
            bh = int(round(board_bbox_img.shape[0] * (bw / max(1, board_bbox_img.shape[1]))))
            board_small = cv2.resize(board_bbox_img, (bw, bh))
            send_img = np.vstack([board_small, warp])
            stages["6_sent_image"] = send_img

            result = _post_image_to_api(_encode_jpeg(send_img, quality=80))

            if not result or not result.get("found"):
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed")
                print("[SCAN] failed: server not found", flush=True)
                time.sleep(0.2)
                continue

            cells = result.get("cells", []) or []
            cam.set_cells_server(cells)

            board_mat = board_from_server_cells(GRID_ROWS, GRID_COLS, cells)
            board.set_board(board_mat)

            cam.set_stage_images(stages)
            cam.set_scan_status("ready")

            print("[SCAN] ok", {"cells": len(cells), "bbox": bbox}, flush=True)
            time.sleep(0.15)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
