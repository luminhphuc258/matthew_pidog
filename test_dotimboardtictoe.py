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
from collections import deque
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file


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

# ---------- Yellow marker HSV ----------
Y_H_MIN = int(os.environ.get("Y_H_MIN", "18"))
Y_H_MAX = int(os.environ.get("Y_H_MAX", "40"))
Y_S_MIN = int(os.environ.get("Y_S_MIN", "70"))
Y_S_MAX = int(os.environ.get("Y_S_MAX", "255"))
Y_V_MIN = int(os.environ.get("Y_V_MIN", "70"))
Y_V_MAX = int(os.environ.get("Y_V_MAX", "255"))

MARKER_MIN_AREA = int(os.environ.get("MARKER_MIN_AREA", "140"))
MARKER_MAX_AREA = int(os.environ.get("MARKER_MAX_AREA", "40000"))
MARKER_MAX_AR = float(os.environ.get("MARKER_MAX_AR", "6.0"))
MARKER_MIN_SIDE = int(os.environ.get("MARKER_MIN_SIDE", "8"))

CORNER_BAND = float(os.environ.get("CORNER_BAND", "0.22"))

# IMPORTANT: không còn reject top nữa, chỉ đánh dấu near_top để debug
TOP_NOISE_REJECT_Y = float(os.environ.get("TOP_NOISE_REJECT_Y", "0.10"))

BBOX_PAD_PX = int(os.environ.get("BBOX_PAD_PX", "0"))
ALLOW_INFER_4TH = str(os.environ.get("ALLOW_INFER_4TH", "1")).lower() in ("1", "true", "yes", "on")
BBOX_SMOOTH_ALPHA = float(os.environ.get("BBOX_SMOOTH_ALPHA", "0.25"))

TOP_X_MATCH_MAX = float(os.environ.get("TOP_X_MATCH_MAX", "0.22"))  # tolerance match theo x

LOG_KEEP = int(os.environ.get("LOG_KEEP", "250"))

# --- NEW: preprocess grid lines before sending to GPT ---
# 0=off, 1=on
ENABLE_GRID_PREPROCESS = str(os.environ.get("ENABLE_GRID_PREPROCESS", "1")).lower() in ("1", "true", "yes", "on")
# adaptive threshold block size (odd)
GRID_ADAPT_BLOCK = int(os.environ.get("GRID_ADAPT_BLOCK", "21"))
GRID_ADAPT_C = int(os.environ.get("GRID_ADAPT_C", "7"))
# line kernel sizes (auto if <=0)
GRID_HK = int(os.environ.get("GRID_HK", "0"))
GRID_VK = int(os.environ.get("GRID_VK", "0"))
# thicken lines a bit
GRID_THICK_DILATE = int(os.environ.get("GRID_THICK_DILATE", "1"))  # 0..2
# remove tiny dots
GRID_DOT_OPEN = int(os.environ.get("GRID_DOT_OPEN", "1"))


# =========================
# Helpers
# =========================
def now_s() -> str:
    return time.strftime("%H:%M:%S")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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
        return {"found": False, "error": f"HTTPError {exc.code}: {detail[:300]}"}
    except urllib.error.URLError as exc:
        return {"found": False, "error": f"URLError: {exc}"}
    except Exception as exc:
        return {"found": False, "error": f"parse failed: {exc}"}


def order_points_4(pts: np.ndarray) -> np.ndarray:
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


def _make_black_warp(text="NO_WARP") -> np.ndarray:
    img = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return img


# =========================
# NEW: Grid preprocess (gray + remove dots + smooth + extract lines)
# =========================
def preprocess_grid_for_gpt(warp_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Output:
      - gray (uint8)
      - bin (0/255)
      - lines (0/255) : combined horizontal+vertical lines
      - final_bgr : white background + black clear grid lines (BGR)
    """
    h, w = warp_bgr.shape[:2]
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)

    # mild denoise (keeps edges)
    gray_blur = cv2.bilateralFilter(gray, 7, 50, 50)

    # adaptive threshold (invert so lines become white)
    blk = GRID_ADAPT_BLOCK
    if blk < 9:
        blk = 9
    if blk % 2 == 0:
        blk += 1

    bin_inv = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blk,
        GRID_ADAPT_C
    )

    # remove tiny dots
    if GRID_DOT_OPEN > 0:
        k = np.ones((3, 3), np.uint8)
        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k, iterations=GRID_DOT_OPEN)

    # line extraction kernels
    hk = GRID_HK if GRID_HK > 0 else max(15, w // 12)   # horizontal kernel length
    vk = GRID_VK if GRID_VK > 0 else max(15, h // 12)   # vertical kernel length

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    horiz = cv2.erode(bin_inv, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=1)

    vert = cv2.erode(bin_inv, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=1)

    lines = cv2.bitwise_or(horiz, vert)

    # connect broken segments a bit
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # thicken lines slightly
    if GRID_THICK_DILATE > 0:
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=GRID_THICK_DILATE)

    # final: white background with black grid lines
    final = 255 - lines  # lines black
    final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    # annotate for debug
    cv2.putText(final_bgr, "grid_clean_gray_lines", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return {
        "gray": gray,
        "bin_inv": bin_inv,
        "lines": lines,
        "final_bgr": final_bgr,
    }


# =========================
# Marker-only Detector (FIX: không reject top markers)
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

        cand: List[Dict[str, Any]] = []
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

            near_top = (cy < TOP_NOISE_REJECT_Y * h)

            box = cv2.boxPoints(rect).astype(np.int32)

            # ✅ FIX: không reject nữa, chỉ đổi màu + label để debug
            if near_top:
                cv2.drawContours(dbg, [box], -1, (0, 165, 255), 2)  # orange
                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 165, 255), -1)
                cv2.putText(dbg, "NEAR_TOP", (int(cx) + 6, int(cy) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.drawContours(dbg, [box], -1, (0, 255, 0), 2)
                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 255, 0), -1)

            cand.append({
                "center": np.array([cx, cy], dtype=np.float32),
                "area": area,
                "ar": float(ar),
                "near_top": bool(near_top),
            })

        # annotate indices
        dbg2 = dbg.copy()
        for i, c in enumerate(cand):
            cx, cy = int(c["center"][0]), int(c["center"][1])
            tag = "T" if c.get("near_top") else " "
            cv2.putText(dbg2, f"#{i}{tag} ({cx},{cy})", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return {"mask": mask, "candidates": cand, "debug": dbg2}

    def _infer_missing_from_3(self, slots: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        keys = {"tl", "tr", "br", "bl"}
        missing = list(keys - set(slots.keys()))
        if len(missing) != 1:
            return None
        m = missing[0]

        tl = slots.get("tl")
        tr = slots.get("tr")
        br = slots.get("br")
        bl = slots.get("bl")

        if bl is not None and br is not None:
            v = br - bl
            if m == "tl" and tr is not None:
                slots["tl"] = tr - v
                return slots
            if m == "tr" and tl is not None:
                slots["tr"] = tl + v
                return slots

        if tl is not None and tr is not None:
            vt = tr - tl
            if m == "bl" and br is not None:
                slots["bl"] = br - vt
                return slots
            if m == "br" and bl is not None:
                slots["br"] = bl + vt
                return slots

        pts = list(slots.values())
        if len(pts) == 3:
            p0, p1, p2 = pts
            slots[m] = p0 + p2 - p1
            return slots
        return None

    def _pick_slots(self, candidates: List[Dict[str, Any]], w: int, h: int) -> Dict[str, Any]:
        if len(candidates) < 2:
            return {"slots": {}, "inferred": False}

        pts = np.array([c["center"] for c in candidates], dtype=np.float32)

        top_y = CORNER_BAND * h
        bot_y = (1.0 - CORNER_BAND) * h
        mid_x = 0.5 * w

        idx_all = list(range(len(pts)))
        idx_bottom = [i for i in idx_all if pts[i][1] >= bot_y]
        idx_top = [i for i in idx_all if pts[i][1] <= top_y]

        # fallback pools
        if len(idx_top) < 2:
            idx_top = sorted(idx_all, key=lambda i: float(pts[i][1]))[:max(2, min(6, len(idx_all)))]
        if len(idx_bottom) < 2:
            idx_bottom = sorted(idx_all, key=lambda i: float(pts[i][1]), reverse=True)[:max(2, min(6, len(idx_all)))]

        def nearest(pool, tx, ty, forbid=set()):
            best_i, best_d = None, 1e18
            for i in pool:
                if i in forbid:
                    continue
                dx = float(pts[i][0] - tx)
                dy = float(pts[i][1] - ty)
                d = dx*dx + dy*dy
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        used = set()
        slots: Dict[str, np.ndarray] = {}

        bl_i = nearest(idx_bottom, 0.0, float(h - 1), forbid=set())
        if bl_i is not None:
            slots["bl"] = pts[bl_i]; used.add(bl_i)

        br_i = nearest(idx_bottom, float(w - 1), float(h - 1), forbid=used)
        if br_i is not None:
            slots["br"] = pts[br_i]; used.add(br_i)

        max_dx = TOP_X_MATCH_MAX * w

        def best_top_match(pool, x_ref, forbid):
            best_i, best_s = None, 1e18
            for i in pool:
                if i in forbid:
                    continue
                x, y = float(pts[i][0]), float(pts[i][1])
                dx = abs(x - float(x_ref))
                if dx > max_dx:
                    continue
                s = (dx*dx) * 2.2 + (y*y) * 0.10
                if s < best_s:
                    best_s = s
                    best_i = i
            return best_i

        # TL x gần BL, TR x gần BR
        if "bl" in slots:
            tl_pool = [i for i in idx_top if pts[i][0] <= mid_x] or idx_top
            tl_i = best_top_match(tl_pool, slots["bl"][0], used)
            if tl_i is not None:
                slots["tl"] = pts[tl_i]; used.add(tl_i)

        if "br" in slots:
            tr_pool = [i for i in idx_top if pts[i][0] >= mid_x] or idx_top
            tr_i = best_top_match(tr_pool, slots["br"][0], used)
            if tr_i is not None:
                slots["tr"] = pts[tr_i]; used.add(tr_i)

        # fallback nếu chỉ thấy top
        if "tl" not in slots:
            tl_i2 = nearest(idx_top, 0.0, 0.0, used)
            if tl_i2 is not None:
                slots["tl"] = pts[tl_i2]; used.add(tl_i2)

        if "tr" not in slots:
            tr_i2 = nearest(idx_top, float(w - 1), 0.0, used)
            if tr_i2 is not None:
                slots["tr"] = pts[tr_i2]; used.add(tr_i2)

        inferred = False
        if ALLOW_INFER_4TH and len(slots) == 3:
            res = self._infer_missing_from_3(slots)
            if res is not None and len(res) == 4:
                inferred = True

        return {"slots": slots, "inferred": inferred}

    def detect_board_bbox(self, frame_bgr) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        ex = self._extract_candidates(frame_bgr)
        cand = ex["candidates"]
        dbg = ex["debug"].copy()
        mask = ex["mask"]

        cv2.putText(dbg, f"cand={len(cand)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)

        picked = self._pick_slots(cand, w, h)
        slots = picked["slots"]
        inferred = picked["inferred"]

        dbg_pick = dbg.copy()
        for k in ["tl", "tr", "br", "bl"]:
            if k in slots:
                p = slots[k]
                cv2.circle(dbg_pick, (int(p[0]), int(p[1])), 10, (255, 0, 255), -1)
                cv2.putText(dbg_pick, k.upper(), (int(p[0]) + 10, int(p[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        cv2.putText(dbg_pick, f"INFER={inferred}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if len(slots) < 4:
            return {
                "found": False,
                "bbox": None,
                "mask": mask,
                "debug": dbg,
                "debug_pick": dbg_pick,
                "inferred": inferred,
                "slots": {k: [float(v[0]), float(v[1])] for k, v in slots.items()},
            }

        centers4 = np.array([slots["tl"], slots["tr"], slots["br"], slots["bl"]], dtype=np.float32)
        centers4 = order_points_4(centers4)

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
            "debug_pick": dbg_pick,
            "board_bbox_img": out,
            "centers4": centers4,
            "inferred": inferred,
            "slots": {k: [float(v[0]), float(v[1])] for k, v in slots.items()},
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
# Web + Camera + Logs
# =========================
class CameraWeb:
    def __init__(self, board: BoardState):
        self.board = board
        self.app = Flask("scan_board_marker_only")
        self._lock = threading.Lock()

        self._last_frame = None
        self._scan_status = "idle"
        self._last_error = ""
        self._cells_server: List[Dict[str, Any]] = []
        self._stages_jpg: Dict[str, bytes] = {}
        self._logs = deque(maxlen=LOG_KEEP)

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
                st["last_error"] = self._last_error
                st["rows"] = self.board.rows
                st["cols"] = self.board.cols
                st["cells_server"] = self._cells_server
                st["stages"] = list(self._stages_jpg.keys())
            st["board"] = self.board.snapshot()
            return jsonify(st)

        @self.app.get("/logs.json")
        def logs():
            with self._lock:
                return jsonify({"lines": list(self._logs)})

        @self.app.get("/play")
        def play():
            self._play_requested.set()
            self.log("[UI] Play Scan clicked")
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

    def log(self, msg: str):
        line = f"{now_s()} {msg}"
        with self._lock:
            self._logs.append(line)
        print(line, flush=True)

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
    .err {{ color:#fca5a5; font-size:12px; white-space:pre-wrap; }}
    .btn {{ background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:8px 12px; border-radius:10px; cursor:pointer; }}
    .video {{ border:1px solid #223; border-radius:10px; width:{cam_w}px; height:{cam_h}px; background:#000; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre; font-size:12px; color:#cbd5f5; }}
    .stages {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; max-height: calc(100vh - 28px); overflow:auto; }}
    .stageimg {{ width:100%; border-radius:10px; border:1px solid #223; margin-top:10px; }}
    .title {{ font-weight:700; margin-bottom:8px; }}
    .muted {{ color:#93a4b8; font-size:12px; }}
    .logs {{ background:#0f172a; border:1px solid #223; border-radius:10px; padding:10px; margin-top:10px; max-height:160px; overflow:auto; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <div class="card">
        <div class="kv"><span class="k">Board:</span> rows=<span id="rows">-</span>, cols=<span id="cols">-</span></div>
        <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">-</span></div>
        <div class="kv"><span class="k">Last error:</span></div>
        <div id="last_error" class="err">-</div>

        <div class="kv"><span class="k">Empty:</span> <span id="empty">-</span></div>
        <div class="kv"><span class="k">Player X:</span> <span id="player_x">-</span></div>
        <div class="kv"><span class="k">Robot O:</span> <span id="robot_o">-</span></div>

        <div class="kv"><span class="k">Board state:</span></div>
        <div id="board" class="mono">-</div>

        <div class="kv"><span class="k">Cells (server):</span></div>
        <div id="cells" class="mono" style="max-height:200px; overflow:auto;">-</div>

        <div class="kv">
          <button class="btn" onclick="playScan()">Play Scan</button>
          <div class="muted" style="margin-top:8px;">NEW: warp -> gray + smooth grid lines -> send to GPT.</div>
        </div>

        <div class="kv"><span class="k">Logs:</span></div>
        <div id="logs" class="logs mono">-</div>
      </div>

      <img class="video" id="cam" src="/mjpeg" />
    </div>

    <div class="stages">
      <div class="title">Processing Stages</div>
      <div class="muted">Always show candidates/picked/warp/grid_clean/would_send.</div>
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
    document.getElementById('last_error').textContent = js.last_error || '-';

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

    const lr = await fetch('/logs.json', {{cache:'no-store'}});
    const ljs = await lr.json();
    document.getElementById('logs').textContent = (ljs.lines || []).slice(-120).join('\\n') || '-';
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

    def set_scan_status(self, status: str, error: str = ""):
        with self._lock:
            self._scan_status = str(status)
            self._last_error = str(error or "")

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
        self.log(f"[CAM] opening {dev}")
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        if not cap.isOpened():
            self.log(f"[CAM] cannot open camera: {dev}")
            self._failed.set()
            self._ready.set()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.log(f"[CAM] opened ok, {CAM_W}x{CAM_H} fps={CAM_FPS}")

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
        self.log(f"[WEB] http://<pi_ip>:{WEB_PORT}/")

    def wait_ready(self, timeout_sec: float = 5.0) -> bool:
        self._ready.wait(timeout=max(0.1, float(timeout_sec)))
        return self._ready.is_set() and not self._failed.is_set()


# =========================
# MAIN
# =========================
def main():
    print("[START] marker_only_board_bbox_send_bbox_and_warp", flush=True)
    board = BoardState(rows=GRID_ROWS, cols=GRID_COLS)
    cam = CameraWeb(board)
    cam.start()

    if not cam.wait_ready(timeout_sec=5.0):
        cam.log("[CAM] not ready, stop")
        return

    detector = MarkerBoardDetector()
    cam.log("[WEB] press Play Scan to run pipeline")

    while True:
        if not cam.consume_scan_request():
            time.sleep(0.08)
            continue

        cam.log("[SCAN] start")
        cam.set_scan_status("scanning", "")

        stages: Dict[str, np.ndarray] = {}
        try:
            frame = cam.get_last_frame()
            if frame is None:
                stages["0_fail"] = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed", "no frame from camera")
                cam.log("[SCAN] failed: no frame")
                continue

            det = detector.detect_board_bbox(frame)
            stages["3m_marker_mask"] = det.get("mask")
            stages["3m_candidates_debug"] = det.get("debug")
            stages["3m_picked_debug"] = det.get("debug_pick", det.get("debug"))

            # Always build "would_send" stages
            board_bbox_img = det.get("board_bbox_img", stages["3m_picked_debug"])
            stages["4_board_bbox"] = board_bbox_img

            if det.get("found") and det.get("bbox") is not None:
                bbox = det["bbox"]
                warp = warp_rect(frame, bbox)
                stages["5_warp"] = warp

                # --- NEW: preprocess warp before sending to GPT ---
                if ENABLE_GRID_PREPROCESS:
                    pp = preprocess_grid_for_gpt(warp)
                    stages["5a_warp_gray"] = pp["gray"]
                    stages["5b_warp_bin_inv"] = pp["bin_inv"]
                    stages["5c_grid_lines"] = pp["lines"]
                    grid_clean = pp["final_bgr"]
                    stages["5d_grid_clean"] = grid_clean
                    warp_for_send = grid_clean
                else:
                    warp_for_send = warp
            else:
                stages["5_warp"] = _make_black_warp("NO_WARP (need >=3 markers)")
                warp_for_send = stages["5_warp"]

            # Stack: board_small + warp_for_send (this is what we actually send)
            bw = warp_for_send.shape[1]
            bh = int(round(board_bbox_img.shape[0] * (bw / max(1, board_bbox_img.shape[1]))))
            board_small = cv2.resize(board_bbox_img, (bw, bh))

            if warp_for_send.shape[:2] != (WARP_H, WARP_W):
                warp_for_send = cv2.resize(warp_for_send, (WARP_W, WARP_H))

            send_img = np.vstack([board_small, warp_for_send])
            stages["6_would_send"] = send_img

            # If cannot get bbox => show stages anyway, but do not send
            if not det.get("found") or det.get("bbox") is None:
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed", "board bbox not found (need >=3 markers)")
                cam.log(f"[SCAN] failed: bbox not found, slots={det.get('slots')}")
                continue

            cam.log("[SCAN] posting to server (grid_clean)...")
            result = _post_image_to_api(_encode_jpeg(send_img, quality=80))

            if not result or not result.get("found"):
                err = (result or {}).get("error", "server returned found=false")
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed", f"server fail: {err}")
                cam.log(f"[SCAN] failed: server: {err}")
                continue

            cells = result.get("cells", []) or []
            cam.set_cells_server(cells)
            board_mat = board_from_server_cells(GRID_ROWS, GRID_COLS, cells)
            board.set_board(board_mat)

            cam.set_stage_images(stages)
            cam.set_scan_status("ready", "")
            cam.log(f"[SCAN] ok: cells={len(cells)} bbox={det['bbox']}")

        except Exception as e:
            cam.set_stage_images(stages or {})
            cam.set_cells_server([])
            cam.set_scan_status("failed", f"exception: {repr(e)}")
            cam.log(f"[SCAN] EXCEPTION: {repr(e)}")

        time.sleep(0.12)


if __name__ == "__main__":
    main()
