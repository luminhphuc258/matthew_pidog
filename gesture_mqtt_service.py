#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ===== chống crash OpenCV/MP khi chạy systemd =====
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import socket
import threading
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from handcommand import HandCommand, HandCfg
from web_dashboard import WebDashboard


# =========================
# ROTATION POLICY
# - Chỉ rotate 1 lần (ở service) để tránh vẽ hand bị trùng
# =========================
ROTATE180 = True


# ===== face UDP =====
FACE_HOST = "127.0.0.1"
FACE_PORT = 39393
_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def set_face(emo: str):
    try:
        _face_sock.sendto(str(emo).encode("utf-8"), (FACE_HOST, FACE_PORT))
    except Exception:
        pass


# =========================
# Snapshot export
# =========================
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "/tmp/gesture_latest.jpg")
SNAPSHOT_JPEG_QUALITY = int(os.environ.get("SNAPSHOT_JPEG_QUALITY", "80"))
SNAPSHOT_INTERVAL_SEC = float(os.environ.get("SNAPSHOT_INTERVAL_SEC", "5.0"))

GESTURE_COOLDOWN_SEC = float(os.environ.get("GESTURE_COOLDOWN_SEC", "0.35"))
GESTURE_RING_MAX = int(os.environ.get("GESTURE_RING_MAX", "80"))

ALWAYS_STOPMUSIC_ON_ANY_GESTURE = os.environ.get("ALWAYS_STOPMUSIC_ON_ANY_GESTURE", "1").strip() == "1"


def _write_snapshot_atomic(frame_bgr, out_path: str, quality: int = 80):
    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp_path, out_path)
    except Exception:
        pass


# =========================
# Shared Camera (V4L2 + MJPG)
# =========================
class Camera:
    def __init__(self, dev="/dev/video0", w=640, h=480, fps=30,
                 snapshot_path: str = SNAPSHOT_PATH,
                 snapshot_interval_sec: float = SNAPSHOT_INTERVAL_SEC,
                 snapshot_quality: int = SNAPSHOT_JPEG_QUALITY):

        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))

        self.last = None
        self.ts = 0.0

        self.snapshot_path = snapshot_path
        self.snapshot_interval_sec = float(snapshot_interval_sec)
        self.snapshot_quality = int(snapshot_quality)
        self._last_snapshot_ts = 0.0

        self._lock = threading.Lock()
        try:
            os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        except Exception:
            pass

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        now = time.time()
        with self._lock:
            self.last = frame
            self.ts = now
            if (now - self._last_snapshot_ts) >= self.snapshot_interval_sec:
                self._last_snapshot_ts = now
                _write_snapshot_atomic(self.last, self.snapshot_path, quality=self.snapshot_quality)
        return frame

    def get_frame(self):
        fr = self.read()
        with self._lock:
            return fr if fr is not None else self.last

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


# ============================================================
# OBSTACLE detector
# - Giữ stripe/hbar rules
# - THÊM rule Edge density + Motion (so với background EMA)
# ============================================================
class ColorStripeObstacleDetector:
    def __init__(self,
                 decision_interval_sec: float = 0.10,
                 window_sec: float = 2.0,
                 roi_y1_ratio: float = 0.10,
                 roi_y2_ratio: float = 0.96,

                 # stripe/hbar params (giữ như bạn)
                 min_vstripe_width_ratio: float = 0.08,
                 max_vstripe_width_ratio: float = 0.70,
                 min_hbar_height_ratio: float = 0.06,
                 max_hbar_height_ratio: float = 0.35,
                 col_std_v_thr: float = 18.0,
                 col_edge_thr: float = 0.040,
                 row_std_v_thr: float = 16.0,
                 row_edge_thr: float = 0.045,
                 stable_mean_delta_thr: float = 10.0,
                 min_good_ratio_in_window: float = 0.65,
                 hbar_min_width_ratio: float = 0.86,
                 neighbor_mean_delta_thr: float = 8.0,

                 # ===== NEW: Edge + Motion =====
                 motion_diff_thr: int = 28,           # pixel diff threshold
                 motion_ratio_thr: float = 0.22,      # tỉ lệ pixel đổi lớn -> obstacle
                 edge_ratio_thr: float = 0.055,       # tỉ lệ edges -> obstacle (case hoạ tiết)
                 bg_alpha: float = 0.05,              # EMA update
                 exposure_jump_v_thr: float = 18.0,   # nếu sáng thay đổi global mạnh -> skip update/bg
                 ):
        self.interval = float(decision_interval_sec)
        self.window_sec = float(window_sec)

        self.ry1 = float(roi_y1_ratio)
        self.ry2 = float(roi_y2_ratio)

        self.min_vw = float(min_vstripe_width_ratio)
        self.max_vw = float(max_vstripe_width_ratio)

        self.min_hh = float(min_hbar_height_ratio)
        self.max_hh = float(max_hbar_height_ratio)

        self.col_std_v_thr = float(col_std_v_thr)
        self.col_edge_thr = float(col_edge_thr)

        self.row_std_v_thr = float(row_std_v_thr)
        self.row_edge_thr = float(row_edge_thr)

        self.stable_mean_delta_thr = float(stable_mean_delta_thr)
        self.min_good_ratio_in_window = float(min_good_ratio_in_window)

        self.hbar_min_width_ratio = float(hbar_min_width_ratio)
        self.neighbor_mean_delta_thr = float(neighbor_mean_delta_thr)

        # NEW
        self.motion_diff_thr = int(motion_diff_thr)
        self.motion_ratio_thr = float(motion_ratio_thr)
        self.edge_ratio_thr = float(edge_ratio_thr)
        self.bg_alpha = float(bg_alpha)
        self.exposure_jump_v_thr = float(exposure_jump_v_thr)

        self._lock = threading.Lock()
        self._last_run = 0.0

        # stripe/hbar history
        self._hist: deque = deque(maxlen=250)

        # motion/edge history by zone: (ts, motion_ratio, edge_ratio)
        self._zone_hist = {
            "LEFT": deque(maxlen=250),
            "CENTER": deque(maxlen=250),
            "RIGHT": deque(maxlen=250),
        }

        # background model (gray float32) for ROI small
        self._bg_gray: Optional[np.ndarray] = None
        self._last_global_v: Optional[float] = None

        self._label = "no obstacle"
        self._zone = "NONE"
        self._orientation = "NONE"
        self._ts = 0.0
        self._shape: Optional[Dict[str, int]] = None
        self._reason: str = ""   # debug: STRIPE / HBAR / MOTION

    @staticmethod
    def _zone_from_x(cx: int, w: int) -> str:
        if cx < w * (1/3):
            return "LEFT"
        if cx > w * (2/3):
            return "RIGHT"
        return "CENTER"

    def _prep_small(self, roi_bgr: np.ndarray, target_w: int = 320) -> Tuple[np.ndarray, float]:
        h, w = roi_bgr.shape[:2]
        if w <= target_w:
            return roi_bgr, 1.0
        scale = target_w / float(w)
        small = cv2.resize(roi_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return small, scale

    def _neighbor_delta_ok_vertical(self, vmap: np.ndarray, x1: int, x2: int) -> bool:
        H, W = vmap.shape[:2]
        band = vmap[:, x1:x2]
        if band.size < 10:
            return False
        m_band = float(np.mean(band))
        pad = max(6, int(0.03 * W))
        xl1 = max(0, x1 - pad)
        xl2 = max(0, x1)
        xr1 = min(W, x2)
        xr2 = min(W, x2 + pad)
        if xl2 - xl1 < 2 or xr2 - xr1 < 2:
            return True
        m_l = float(np.mean(vmap[:, xl1:xl2]))
        m_r = float(np.mean(vmap[:, xr1:xr2]))
        if (abs(m_band - m_l) < self.neighbor_mean_delta_thr) and (abs(m_band - m_r) < self.neighbor_mean_delta_thr):
            return False
        return True

    def _neighbor_delta_ok_horizontal(self, vmap: np.ndarray, y1: int, y2: int) -> bool:
        H, W = vmap.shape[:2]
        band = vmap[y1:y2, :]
        if band.size < 10:
            return False
        m_band = float(np.mean(band))
        pad = max(6, int(0.03 * H))
        yu1 = max(0, y1 - pad)
        yu2 = max(0, y1)
        yd1 = min(H, y2)
        yd2 = min(H, y2 + pad)
        if yu2 - yu1 < 2 or yd2 - yd1 < 2:
            return True
        m_u = float(np.mean(vmap[yu1:yu2, :]))
        m_d = float(np.mean(vmap[yd1:yd2, :]))
        if (abs(m_band - m_u) < self.neighbor_mean_delta_thr) and (abs(m_band - m_d) < self.neighbor_mean_delta_thr):
            return False
        return True

    def _find_vertical_stripe(self, roi_bgr: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        h, w = roi_bgr.shape[:2]
        if w < 50 or h < 50:
            return False, None, 0.0

        small, scale = self._prep_small(roi_bgr, target_w=320)
        hh, ww = small.shape[:2]

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edges = (edges > 0).astype(np.float32)

        col_std_v = np.std(v, axis=0)
        col_edge = np.mean(edges, axis=0)

        good = (col_std_v < self.col_std_v_thr) & (col_edge < self.col_edge_thr)

        best_len = 0
        best_run = None
        i = 0
        while i < ww:
            if not good[i]:
                i += 1
                continue
            j = i
            while j < ww and good[j]:
                j += 1
            run_len = j - i
            if run_len > best_len:
                best_len = run_len
                best_run = (i, j)
            i = j

        if best_run is None:
            return False, None, 0.0

        min_w = int(self.min_vw * ww)
        max_w = int(self.max_vw * ww)
        if best_len < max(10, min_w) or best_len > max_w:
            return False, None, 0.0

        x1s, x2s = best_run
        meanV = float(np.mean(v[:, x1s:x2s]))

        if not self._neighbor_delta_ok_vertical(v, x1s, x2s):
            return False, None, 0.0

        if scale != 1.0:
            x1 = int(x1s / scale)
            x2 = int(x2s / scale)
        else:
            x1, x2 = int(x1s), int(x2s)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        if x2 - x1 < int(self.min_vw * w):
            return False, None, 0.0

        return True, (x1, x2), meanV

    def _find_horizontal_bar(self, roi_bgr: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        h, w = roi_bgr.shape[:2]
        if w < 80 or h < 50:
            return False, None, 0.0

        small, scale = self._prep_small(roi_bgr, target_w=320)
        hh, ww = small.shape[:2]

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edges = (edges > 0).astype(np.float32)

        row_std_v = np.std(v, axis=1)
        row_edge = np.mean(edges, axis=1)

        good = (row_std_v < self.row_std_v_thr) & (row_edge < self.row_edge_thr)

        best_len = 0
        best_run = None
        i = 0
        while i < hh:
            if not good[i]:
                i += 1
                continue
            j = i
            while j < hh and good[j]:
                j += 1
            run_len = j - i
            if run_len > best_len:
                best_len = run_len
                best_run = (i, j)
            i = j

        if best_run is None:
            return False, None, 0.0

        min_h = int(self.min_hh * hh)
        max_h = int(self.max_hh * hh)
        if best_len < max(8, min_h) or best_len > max_h:
            return False, None, 0.0

        y1s, y2s = best_run

        mid_x1 = int(0.05 * ww)
        mid_x2 = int(0.95 * ww)
        if (mid_x2 - mid_x1) / float(ww) < self.hbar_min_width_ratio:
            return False, None, 0.0

        meanV = float(np.mean(v[y1s:y2s, :]))

        if not self._neighbor_delta_ok_horizontal(v, y1s, y2s):
            return False, None, 0.0

        if scale != 1.0:
            y1 = int(y1s / scale)
            y2 = int(y2s / scale)
        else:
            y1, y2 = int(y1s), int(y2s)

        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if y2 - y1 < int(self.min_hh * h):
            return False, None, 0.0

        return True, (y1, y2), meanV

    def _global_exposure_jump(self, roi_bgr_small: np.ndarray) -> bool:
        try:
            hsv = cv2.cvtColor(roi_bgr_small, cv2.COLOR_BGR2HSV)
            vmean = float(np.mean(hsv[:, :, 2]))
        except Exception:
            return False

        if self._last_global_v is None:
            self._last_global_v = vmean
            return False

        dv = abs(vmean - self._last_global_v)
        self._last_global_v = vmean
        return dv >= self.exposure_jump_v_thr

    def _update_motion_edge(self, roi_bgr: np.ndarray, now: float):
        """
        Compute motion vs background (EMA) + edge density for 3 zones.
        Lưu hist trong window_sec để quyết định.
        """
        small, _ = self._prep_small(roi_bgr, target_w=320)
        jump = self._global_exposure_jump(small)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._bg_gray is None or self._bg_gray.shape != gray.shape:
            self._bg_gray = gray.copy()

        # compute edge ratio using Canny on uint8
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        edges = cv2.Canny(g8, 60, 160)
        edges01 = (edges > 0).astype(np.float32)

        Hs, Ws = gray.shape[:2]
        zW = Ws // 3
        zones = {
            "LEFT": (0, zW),
            "CENTER": (zW, 2 * zW),
            "RIGHT": (2 * zW, Ws),
        }

        # motion vs bg
        diff = np.abs(gray - self._bg_gray)
        motion_mask = (diff >= float(self.motion_diff_thr)).astype(np.float32)

        # Update history per zone
        for zn, (x1, x2) in zones.items():
            m = motion_mask[:, x1:x2]
            e = edges01[:, x1:x2]
            motion_ratio = float(np.mean(m)) if m.size else 0.0
            edge_ratio = float(np.mean(e)) if e.size else 0.0
            self._zone_hist[zn].append((now, motion_ratio, edge_ratio))

        # Update bg if not exposure jump and scene seems stable (no big motion overall)
        # (chỉ update khi motion toàn ROI nhỏ)
        if not jump:
            overall_motion = float(np.mean(motion_mask)) if motion_mask.size else 0.0
            # chỉ update bg khi khá yên (không có vật cản lớn)
            if overall_motion < (self.motion_ratio_thr * 0.55):
                a = self.bg_alpha
                self._bg_gray = (1.0 - a) * self._bg_gray + a * gray

    def _decide_motion_zone(self, now: float) -> Optional[Tuple[str, float, float]]:
        """
        Quyết định obstacle theo rule (Motion + Edge) trong window_sec.
        Return: (zone, avg_motion, avg_edge) or None
        """
        tmin = now - self.window_sec
        best = None

        for zn, dq in self._zone_hist.items():
            items = [it for it in dq if it[0] >= tmin]
            if len(items) < 6:
                continue
            good = [it for it in items if (it[1] >= self.motion_ratio_thr and it[2] >= self.edge_ratio_thr)]
            if len(good) / float(len(items)) < self.min_good_ratio_in_window:
                continue

            avg_m = float(np.mean([it[1] for it in items]))
            avg_e = float(np.mean([it[2] for it in items]))

            # chọn zone mạnh nhất theo motion trước, rồi edge
            cand = (zn, avg_m, avg_e)
            if best is None:
                best = cand
            else:
                if (cand[1] > best[1]) or (cand[1] == best[1] and cand[2] > best[2]):
                    best = cand

        return best

    def compute(self, frame_bgr: np.ndarray):
        now = time.time()
        if (now - self._last_run) < self.interval:
            return
        self._last_run = now

        H, W = frame_bgr.shape[:2]
        ry1 = int(H * self.ry1)
        ry2 = int(H * self.ry2)
        ry1 = max(0, min(H - 1, ry1))
        ry2 = max(ry1 + 10, min(H, ry2))

        roi = frame_bgr[ry1:ry2, :]

        # update motion/edge first
        self._update_motion_edge(roi, now)

        # detect stripe/hbar
        vok, vrun, vmean = self._find_vertical_stripe(roi)
        vx1 = vx2 = -1
        if vok and vrun:
            vx1, vx2 = vrun[0], vrun[1]

        hok, hrun, hmean = self._find_horizontal_bar(roi)
        hy1 = hy2 = -1
        if hok and hrun:
            hy1, hy2 = hrun[0], hrun[1]

        # store stripe/hbar history
        self._hist.append((now, bool(vok), "VERTICAL", vx1 if vok else -1, vx2 if vok else -1, ry1, ry2, vmean if vok else 0.0))
        self._hist.append((now, bool(hok), "HORIZONTAL", 0 if hok else -1, W if hok else -1, (ry1 + hy1) if hok else -1, (ry1 + hy2) if hok else -1, hmean if hok else 0.0))

        # decide stripe/hbar in window
        tmin = now - self.window_sec
        items = [it for it in list(self._hist) if it[0] >= tmin]
        v_items = [it for it in items if it[2] == "VERTICAL"]
        h_items = [it for it in items if it[2] == "HORIZONTAL"]

        def decide(items_oriented: List[tuple]) -> Optional[tuple]:
            if len(items_oriented) < 6:
                return None
            oks = [it for it in items_oriented if it[1]]
            if len(oks) / float(len(items_oriented)) < self.min_good_ratio_in_window:
                return None
            vs = [it[7] for it in oks]
            if not vs or (max(vs) - min(vs)) > self.stable_mean_delta_thr:
                return None
            oks_sorted = sorted(oks, key=lambda x: (x[4] - x[3]) * max(1, (x[6] - x[5])), reverse=True)
            return oks_sorted[0] if oks_sorted else None

        best_v = decide(v_items)
        best_h = decide(h_items)

        chosen = None
        reason = ""

        # ưu tiên hbar full width
        if best_h and (best_h[4] - best_h[3]) >= int(0.85 * W):
            chosen = best_h
            reason = "HBAR"
        elif best_v:
            chosen = best_v
            reason = "STRIPE"
        elif best_h:
            chosen = best_h
            reason = "HBAR"

        # nếu stripe/hbar fail -> dùng motion+edge
        motion_pick = self._decide_motion_zone(now)
        if chosen is None and motion_pick is not None:
            zn, avg_m, avg_e = motion_pick
            # shape là full-height zone stripe
            if zn == "LEFT":
                x1, x2 = 0, W // 3
            elif zn == "CENTER":
                x1, x2 = W // 3, 2 * (W // 3)
            else:
                x1, x2 = 2 * (W // 3), W

            with self._lock:
                self._label = "yes have obstacle"
                self._zone = zn
                self._orientation = "VERTICAL"
                self._ts = now
                self._shape = {"x1": int(x1), "x2": int(x2), "y1": int(ry1), "y2": int(ry2)}
                self._reason = f"MOTION(m={avg_m:.2f},e={avg_e:.2f})"
            return

        if not chosen:
            with self._lock:
                self._label = "no obstacle"
                self._zone = "NONE"
                self._orientation = "NONE"
                self._ts = now
                self._shape = None
                self._reason = ""
            return

        _, _, ori, x1, x2, y1, y2, _ = chosen

        if ori == "VERTICAL":
            cx = int((x1 + x2) / 2)
            zone = self._zone_from_x(cx, W)
            shape = {"x1": int(x1), "x2": int(x2), "y1": int(ry1), "y2": int(ry2)}
        else:
            zone = "CENTER"
            shape = {"x1": int(x1), "x2": int(x2), "y1": int(y1), "y2": int(y2)}

        with self._lock:
            self._label = "yes have obstacle"
            self._zone = zone
            self._orientation = ori
            self._ts = now
            self._shape = shape
            self._reason = reason

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "label": self._label,
                "zone": self._zone,
                "orientation": self._orientation,
                "ts": self._ts,
                "shape": self._shape,
                "reason": self._reason,
            }

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        st = self.get_state()
        out = frame_bgr.copy()
        if st.get("label") != "yes have obstacle":
            return out
        shp = st.get("shape")
        if not isinstance(shp, dict):
            return out

        x1 = int(shp.get("x1", 0)); x2 = int(shp.get("x2", 0))
        y1 = int(shp.get("y1", 0)); y2 = int(shp.get("y2", 0))

        x1 = max(0, min(out.shape[1] - 1, x1))
        x2 = max(0, min(out.shape[1], x2))
        y1 = max(0, min(out.shape[0] - 1, y1))
        y2 = max(0, min(out.shape[0], y2))

        if x2 > x1 and y2 > y1:
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return out


# =========================
# Gesture store
# =========================
gesture_lock = threading.Lock()
gesture_latest: Dict[str, Any] = {"label": "", "ts": 0.0, "face": "", "raw": ""}
gesture_ring = deque(maxlen=GESTURE_RING_MAX)
_last_gesture_ts_by_label: Dict[str, float] = {}

def _push_gesture(label: str, face: str = "", raw: str = ""):
    now = time.time()
    with gesture_lock:
        last = _last_gesture_ts_by_label.get(label, 0.0)
        if (now - last) < GESTURE_COOLDOWN_SEC:
            return
        _last_gesture_ts_by_label[label] = now
        gesture_latest.update({"label": label, "ts": now, "face": face or "", "raw": raw or ""})
        gesture_ring.append({"label": label, "ts": now, "face": face or "", "raw": raw or ""})


# =========================
# Hand overlay (draw_on_frame) + FIX duplicate top/bottom
# =========================
def _hand_enabled(hc: HandCommand) -> bool:
    try:
        st = hc.get_status() if hasattr(hc, "get_status") else {}
        if isinstance(st, dict) and "enabled" in st:
            return bool(st.get("enabled", False))
    except Exception:
        pass
    for attr in ("enabled", "is_enabled", "_enabled"):
        if hasattr(hc, attr):
            try:
                v = getattr(hc, attr)
                if isinstance(v, bool):
                    return v
            except Exception:
                pass
    return True

def _looks_like_black_banner(img: np.ndarray, y1: int, y2: int) -> bool:
    try:
        strip = img[y1:y2, :, :]
        if strip.size < 10:
            return False
        # banner đen: mean thấp + khá đều
        m = float(np.mean(strip))
        return m < 35.0
    except Exception:
        return False

def _fix_double_banner(base_bgr: np.ndarray, drawn_bgr: np.ndarray) -> np.ndarray:
    """
    Nếu có banner đen ở TOP và BOTTOM cùng lúc => xoá banner dưới bằng cách restore từ base.
    """
    if base_bgr is None or drawn_bgr is None:
        return drawn_bgr

    h = drawn_bgr.shape[0]
    band = max(44, int(0.10 * h))  # ~10% height

    top_black = _looks_like_black_banner(drawn_bgr, 0, band)
    bot_black = _looks_like_black_banner(drawn_bgr, h - band, h)

    if top_black and bot_black:
        out = drawn_bgr.copy()
        out[h - band:h, :, :] = base_bgr[h - band:h, :, :]
        return out

    return drawn_bgr

def _draw_hand_old_style(hc: HandCommand, frame_bgr: np.ndarray) -> np.ndarray:
    if hc is None or not _hand_enabled(hc):
        return frame_bgr

    base = frame_bgr
    if hasattr(hc, "draw_on_frame"):
        try:
            out = hc.draw_on_frame(frame_bgr.copy())
            if isinstance(out, np.ndarray) and out.shape[:2] == frame_bgr.shape[:2]:
                out = _fix_double_banner(base, out)
                return out
        except Exception:
            pass

    # fallback text
    out = frame_bgr
    try:
        st = hc.get_status() if hasattr(hc, "get_status") else {}
        if isinstance(st, dict):
            en = st.get("enabled", False)
            act = st.get("action", None)
            fps = st.get("fps", None)
            fc = st.get("finger_count", None)
            if isinstance(fps, (int, float)):
                txt = f"HAND: {'ON' if en else 'OFF'}  {act if act else 'NA'}  fingers:{fc if fc is not None else 'NA'}  ({fps:.1f}fps)"
            else:
                txt = f"HAND: {'ON' if en else 'OFF'}  {act if act else 'NA'}  fingers:{fc if fc is not None else 'NA'}"
            cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    except Exception:
        pass

    return out


def _route_exists(app, rule: str) -> bool:
    try:
        for r in app.url_map.iter_rules():
            if str(r.rule) == str(rule):
                return True
    except Exception:
        pass
    return False


def _rot_if_needed(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return None
    if ROTATE180:
        try:
            return cv2.rotate(frame, cv2.ROTATE_180)
        except Exception:
            return frame
    return frame


def main():
    cam = Camera(dev="/dev/video0", w=640, h=480, fps=30,
                 snapshot_path=SNAPSHOT_PATH,
                 snapshot_interval_sec=SNAPSHOT_INTERVAL_SEC,
                 snapshot_quality=SNAPSHOT_JPEG_QUALITY)

    detector = ColorStripeObstacleDetector(
        # bạn có thể tweak thêm nếu muốn
        motion_diff_thr=28,
        motion_ratio_thr=0.22,
        edge_ratio_thr=0.055,
        bg_alpha=0.05,
        exposure_jump_v_thr=18.0
    )

    # FIX remap labels:
    ACTION_REMAP = {"UP": "SIT", "SIT": "STANDUP"}

    def on_action(action: str, face: str, bark: bool):
        raw = (action or "").strip().upper()
        if not raw:
            return
        mapped = ACTION_REMAP.get(raw, raw)
        print(f"[GEST] raw={raw} -> {mapped}" if mapped != raw else f"[GEST] {mapped}", flush=True)

        if face:
            set_face(face)

        _push_gesture(mapped, face=face or "", raw=raw)

        if ALWAYS_STOPMUSIC_ON_ANY_GESTURE:
            _push_gesture("STOPMUSIC", face=face or "", raw=f"auto_from_{mapped}")

    # ===== shared camera frame provider (rotated) =====
    def get_frame_rotated():
        fr = cam.get_frame()
        return _rot_if_needed(fr)

    # HandCommand KHÔNG mở camera lần 2, chỉ lấy frame từ callback
    hc = HandCommand(
        cfg=HandCfg(
            cam_dev="",  # tránh mở camera lần 2
            w=640, h=480, fps=30,
            process_every=2,
            action_cooldown_sec=0.7,
            pos_left_x=0.18,
            pos_right_x=0.82,
            pos_up_y=0.22,
            pos_down_y=0.78,
            pos_hold_frames=4,
        ),
        on_action=on_action,
        boot_helper=None,
        get_frame_bgr=get_frame_rotated,   # ✅ feed rotated frame để overlay khớp camera view
        open_own_camera=False,
        clear_memory_on_start=True
    )

    hc.start()
    hc.set_enabled(True)

    def get_frame_for_dashboard():
        frame = get_frame_rotated()
        if frame is None:
            return None

        detector.compute(frame)
        out = detector.draw_overlay(frame)

        # vẽ hand (fix duplicate)
        out = _draw_hand_old_style(hc, out)
        return out

    # IMPORTANT: rotate180=False vì mình rotate ở service rồi (tránh double rotate)
    dash = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_frame_bgr=get_frame_for_dashboard,
        get_obstacle_state=lambda: detector.get_state(),
        on_manual_cmd=lambda m: print("[MANUAL CMD]", m, flush=True),
        rotate180=False,  # ✅ FIX trùng overlay
        hand_command=hc,
    )

    app = dash.app

    # add routes only if missing (tránh overwrite -> ABRT)
    if not _route_exists(app, "/take_gesture_meaning"):
        @app.get("/take_gesture_meaning")
        def take_gesture_meaning():
            from flask import jsonify, request
            since_ts = request.args.get("since_ts", None)
            limit = request.args.get("limit", None)

            try:
                since_ts_f = float(since_ts) if since_ts is not None else None
            except Exception:
                since_ts_f = None
            try:
                limit_n = int(limit) if limit is not None else 20
            except Exception:
                limit_n = 20

            with gesture_lock:
                latest = dict(gesture_latest)
                events = list(gesture_ring)

            if since_ts_f is not None:
                events = [e for e in events if float(e.get("ts", 0.0)) > since_ts_f]
            events = events[-max(1, min(200, limit_n)):]
            return jsonify({"ok": True, "latest": latest, "events": events})

    if not _route_exists(app, "/take_camera_decision"):
        @app.get("/take_camera_decision")
        def take_camera_decision():
            from flask import jsonify
            st = detector.get_state()
            return jsonify({"ok": True, **st})

    print("\n=== Gesture Service + WebDashboard (SINGLE PORT 8000) ===", flush=True)
    print("WebDashboard: http://<pi_ip>:8000", flush=True)
    print("Gesture API : http://<pi_ip>:8000/take_gesture_meaning", flush=True)
    print("Camera API  : http://<pi_ip>:8000/take_camera_decision", flush=True)
    print("Snapshot    :", SNAPSHOT_PATH, flush=True)
    print("Rotate180   :", ROTATE180, flush=True)

    try:
        dash.run()
    finally:
        try:
            hc.stop()
        except Exception:
            pass
        cam.close()


if __name__ == "__main__":
    main()
