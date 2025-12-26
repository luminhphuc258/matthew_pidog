#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import socket
import threading
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

from handcommand import HandCommand, HandCfg
from web_dashboard import WebDashboard


# ===== face UDP (giống face3d) =====
FACE_HOST = "127.0.0.1"
FACE_PORT = 39393
_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def set_face(emo: str):
    try:
        _face_sock.sendto(emo.encode("utf-8"), (FACE_HOST, FACE_PORT))
    except Exception:
        pass


# =========================
# Snapshot export
# =========================
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "/tmp/gesture_latest.jpg")
SNAPSHOT_JPEG_QUALITY = int(os.environ.get("SNAPSHOT_JPEG_QUALITY", "80"))
SNAPSHOT_INTERVAL_SEC = float(os.environ.get("SNAPSHOT_INTERVAL_SEC", "5.0"))

# Gesture internal publish
GESTURE_COOLDOWN_SEC = float(os.environ.get("GESTURE_COOLDOWN_SEC", "0.35"))
GESTURE_RING_MAX = int(os.environ.get("GESTURE_RING_MAX", "80"))

# Any gesture -> STOPMUSIC internal event
ALWAYS_STOPMUSIC_ON_ANY_GESTURE = os.environ.get("ALWAYS_STOPMUSIC_ON_ANY_GESTURE", "1").strip() == "1"


def _write_snapshot_atomic(frame_bgr, out_path: str, quality: int = 80):
    """Ghi JPEG atomic để process khác đọc không bị file hỏng."""
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
# Shared Camera
# =========================
class Camera:
    def __init__(self, dev="/dev/video0", w=640, h=480, fps=30,
                 snapshot_path: str = SNAPSHOT_PATH,
                 snapshot_interval_sec: float = SNAPSHOT_INTERVAL_SEC,
                 snapshot_quality: int = SNAPSHOT_JPEG_QUALITY):
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

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
# OBSTACLE by COLOR CONSISTENCY (2s) + FULL-HEIGHT STRIPE
# + NEW: HORIZONTAL BAR (almost full width) -> obstacle too
# ============================================================
class ColorStripeObstacleDetector:
    def __init__(self,
                 decision_interval_sec: float = 0.10,
                 window_sec: float = 2.0,
                 roi_y1_ratio: float = 0.10,
                 roi_y2_ratio: float = 0.96,

                 # vertical stripe width limits
                 min_vstripe_width_ratio: float = 0.08,
                 max_vstripe_width_ratio: float = 0.70,

                 # horizontal bar thickness limits (as ratio of height ROI)
                 min_hbar_height_ratio: float = 0.06,
                 max_hbar_height_ratio: float = 0.35,

                 # thresholds for "uniform"
                 col_std_v_thr: float = 18.0,
                 col_edge_thr: float = 0.040,

                 row_std_v_thr: float = 16.0,
                 row_edge_thr: float = 0.045,

                 # stable within window
                 stable_mean_delta_thr: float = 10.0,
                 min_good_ratio_in_window: float = 0.65,

                 # NEW: full-width horizontal bar rule
                 hbar_min_width_ratio: float = 0.86,  # gần full ngang -> coi là "ống nằm ngang"
                 neighbor_mean_delta_thr: float = 8.0 # band phải khác nền xung quanh để tránh false positive
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

        self._lock = threading.Lock()
        self._last_run = 0.0

        # rolling window history
        # item: (ts, ok, orientation, x1,x2,y1,y2, meanV)
        self._hist: deque = deque(maxlen=250)

        self._label = "no obstacle"
        self._zone = "NONE"
        self._orientation = "NONE"
        self._ts = 0.0
        self._shape: Optional[Dict[str, int]] = None  # box coords

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
        small = cv2.resize(roi_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return small, scale

    def _neighbor_delta_ok_vertical(self, vmap: np.ndarray, x1: int, x2: int) -> bool:
        # vmap shape (H,W) float32
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
            return True  # cannot compare -> allow
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

        # neighbor delta filter to avoid background smooth region
        if not self._neighbor_delta_ok_vertical(v, x1s, x2s):
            return False, None, 0.0

        # map back
        if scale != 1.0:
            x1 = int(x1s / scale)
            x2 = int(x2s / scale)
        else:
            x1, x2 = int(x1s), int(x2s)

        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w, x2))
        if x2 - x1 < int(self.min_vw * w):
            return False, None, 0.0

        return True, (x1, x2), meanV

    def _find_horizontal_bar(self, roi_bgr: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        """
        NEW RULE:
        - find contiguous rows where each row is uniform across width (low std + low edges)
        - band must span almost full width (we measure on full row already)
        - band thickness within [min_hh, max_hh]
        - band must differ from neighbors (above/below) to avoid static background
        """
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

        # per-row stats
        row_std_v = np.std(v, axis=1)      # (hh,)
        row_edge = np.mean(edges, axis=1)  # (hh,)

        good = (row_std_v < self.row_std_v_thr) & (row_edge < self.row_edge_thr)

        # longest run
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

        # horizontal bar width rule: nearly full width
        # (we already use full row), but still avoid cases where only center part uniform:
        # check uniformity also on 90% middle width segment
        mid_x1 = int(0.05 * ww)
        mid_x2 = int(0.95 * ww)
        band_mid = v[y1s:y2s, mid_x1:mid_x2]
        if band_mid.size < 10:
            return False, None, 0.0

        # require that mid segment width ratio passes
        if (mid_x2 - mid_x1) / float(ww) < self.hbar_min_width_ratio:
            return False, None, 0.0

        meanV = float(np.mean(v[y1s:y2s, :]))

        # neighbor delta filter
        if not self._neighbor_delta_ok_horizontal(v, y1s, y2s):
            return False, None, 0.0

        # map back
        if scale != 1.0:
            y1 = int(y1s / scale)
            y2 = int(y2s / scale)
        else:
            y1, y2 = int(y1s), int(y2s)

        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h, y2))
        if y2 - y1 < int(self.min_hh * h):
            return False, None, 0.0

        return True, (y1, y2), meanV

    def compute(self, frame_bgr: np.ndarray):
        now = time.time()
        if (now - self._last_run) < self.interval:
            return
        self._last_run = now

        H, W = frame_bgr.shape[:2]
        ry1 = int(H * self.ry1)
        ry2 = int(H * self.ry2)
        ry1 = max(0, min(H-1, ry1))
        ry2 = max(ry1+10, min(H, ry2))

        roi = frame_bgr[ry1:ry2, :]

        # detect vertical stripe
        vok, vrun, vmean = self._find_vertical_stripe(roi)
        vx1 = vx2 = -1
        if vok and vrun:
            vx1, vx2 = vrun[0], vrun[1]

        # detect horizontal bar
        hok, hrun, hmean = self._find_horizontal_bar(roi)
        hy1 = hy2 = -1
        if hok and hrun:
            hy1, hy2 = hrun[0], hrun[1]

        # push history items (prefer the stronger one later)
        if vok:
            self._hist.append((now, True, "VERTICAL", vx1, vx2, ry1, ry2, vmean))
        else:
            self._hist.append((now, False, "VERTICAL", -1, -1, ry1, ry2, 0.0))

        if hok:
            # store horizontal shape with x spanning full width
            self._hist.append((now, True, "HORIZONTAL", 0, W, ry1 + hy1, ry1 + hy2, hmean))
        else:
            self._hist.append((now, False, "HORIZONTAL", -1, -1, -1, -1, 0.0))

        # evaluate window
        tmin = now - self.window_sec
        items = [it for it in list(self._hist) if it[0] >= tmin]

        # split by orientation
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
            # pick widest/strongest
            oks_sorted = sorted(oks, key=lambda x: (x[4]-x[3]) * (x[6]-x[5]), reverse=True)
            return oks_sorted[0] if oks_sorted else None

        best_v = decide(v_items)
        best_h = decide(h_items)

        # choose best: prefer horizontal if it spans almost full width (x2-x1 big)
        chosen = None
        if best_h and (best_h[4] - best_h[3]) >= int(0.85 * W):
            chosen = best_h
        elif best_v:
            chosen = best_v
        elif best_h:
            chosen = best_h

        if not chosen:
            with self._lock:
                self._label = "no obstacle"
                self._zone = "NONE"
                self._orientation = "NONE"
                self._ts = now
                self._shape = None
            return

        _, _, ori, x1, x2, y1, y2, _ = chosen

        if ori == "VERTICAL":
            cx = int((x1 + x2) / 2)
            zone = self._zone_from_x(cx, W)
            shape = {"x1": int(x1), "x2": int(x2), "y1": int(ry1), "y2": int(ry2)}
        else:
            # horizontal: zone keep CENTER (spans width)
            zone = "CENTER"
            shape = {"x1": int(x1), "x2": int(x2), "y1": int(y1), "y2": int(y2)}

        with self._lock:
            self._label = "yes have obstacle"
            self._zone = zone
            self._orientation = ori
            self._ts = now
            self._shape = shape

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "label": self._label,
                "zone": self._zone,
                "orientation": self._orientation,
                "ts": self._ts,
                "shape": self._shape
            }

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        st = self.get_state()
        out = frame_bgr.copy()

        if st.get("label") != "yes have obstacle":
            return out

        shp = st.get("shape")
        if not isinstance(shp, dict):
            return out

        x1 = int(shp.get("x1", 0))
        x2 = int(shp.get("x2", 0))
        y1 = int(shp.get("y1", 0))
        y2 = int(shp.get("y2", 0))

        x1 = max(0, min(out.shape[1]-1, x1))
        x2 = max(0, min(out.shape[1], x2))
        y1 = max(0, min(out.shape[0]-1, y1))
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
# Hand overlay helper (restore drawing)
# =========================
def _try_hand_draw(hc: HandCommand, frame_bgr: np.ndarray) -> np.ndarray:
    """
    Cố gắng dùng các hàm draw/annotate có sẵn trong HandCommand.
    Nếu không có thì fallback vẽ text từ status.
    """
    if hc is None:
        return frame_bgr

    # 1) Try known methods (tùy code bạn đặt tên gì)
    for meth in ("get_debug_draw", "draw_debug", "annotate", "draw_overlay", "render_debug"):
        if hasattr(hc, meth):
            fn = getattr(hc, meth)
            try:
                out = fn(frame_bgr)
                if isinstance(out, np.ndarray) and out.shape[:2] == frame_bgr.shape[:2]:
                    return out
            except Exception:
                pass

    # 2) Fallback: write status text
    out = frame_bgr
    try:
        st = hc.get_status() if hasattr(hc, "get_status") else {}
        if isinstance(st, dict):
            en = st.get("enabled", False)
            act = st.get("action", None)
            fps = st.get("fps", None)
            fc = st.get("finger_count", None)

            txt = f"HAND: {'ON' if en else 'OFF'}  {act if act else 'NA'}  fingers:{fc if fc is not None else 'NA'}  ({fps:.1f}fps)" if isinstance(fps, (int, float)) else \
                  f"HAND: {'ON' if en else 'OFF'}  {act if act else 'NA'}  fingers:{fc if fc is not None else 'NA'}"
            cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    except Exception:
        pass

    return out


# =========================
# Main
# =========================
def main():
    cam = Camera(dev="/dev/video0", w=640, h=480, fps=30,
                 snapshot_path=SNAPSHOT_PATH,
                 snapshot_interval_sec=SNAPSHOT_INTERVAL_SEC,
                 snapshot_quality=SNAPSHOT_JPEG_QUALITY)

    detector = ColorStripeObstacleDetector(
        decision_interval_sec=0.10,
        window_sec=2.0,
        roi_y1_ratio=0.10,
        roi_y2_ratio=0.96,

        min_vstripe_width_ratio=0.08,
        max_vstripe_width_ratio=0.70,

        min_hbar_height_ratio=0.06,
        max_hbar_height_ratio=0.35,

        col_std_v_thr=18.0,
        col_edge_thr=0.040,

        row_std_v_thr=16.0,
        row_edge_thr=0.045,

        stable_mean_delta_thr=10.0,
        min_good_ratio_in_window=0.65,

        hbar_min_width_ratio=0.86,
        neighbor_mean_delta_thr=8.0
    )

    # =========================
    # FIX remap labels:
    # UP -> SIT
    # SIT -> STANDUP
    # =========================
    ACTION_REMAP = {
        "UP": "SIT",
        "SIT": "STANDUP",
    }

    def on_action(action: str, face: str, bark: bool):
        raw = (action or "").strip().upper()
        if not raw:
            return

        mapped = ACTION_REMAP.get(raw, raw)

        if mapped != raw:
            print(f"[GEST] raw={raw} -> mapped={mapped}", flush=True)
        else:
            print(f"[GEST] {mapped}", flush=True)

        if face:
            set_face(face)

        _push_gesture(mapped, face=face or "", raw=raw)

        if ALWAYS_STOPMUSIC_ON_ANY_GESTURE:
            _push_gesture("STOPMUSIC", face=face or "", raw=f"auto_from_{mapped}")

    # HandCommand uses raw frame
    def get_frame_raw():
        return cam.get_frame()

    hc = HandCommand(
        cfg=HandCfg(
            cam_dev="/dev/video0",
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
        get_frame_bgr=get_frame_raw,
        open_own_camera=False,
        clear_memory_on_start=True
    )

    hc.start()
    hc.set_enabled(True)

    # Dashboard frame: obstacle + hand overlay together
    def get_frame_for_dashboard():
        frame = cam.get_frame()
        if frame is None:
            return None

        detector.compute(frame)
        out = detector.draw_overlay(frame)

        # restore hand drawing overlay
        out = _try_hand_draw(hc, out)
        return out

    dash = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_frame_bgr=get_frame_for_dashboard,
        get_obstacle_state=lambda: detector.get_state(),
        on_manual_cmd=lambda m: print("[MANUAL CMD]", m, flush=True),
        rotate180=True,
        hand_command=hc,
    )

    # =========================
    # ADD API ROUTES into same PORT 8000
    # =========================
    app = dash.app  # Flask instance from WebDashboard

    @app.get("/take_gesture_meaning")
    def take_gesture_meaning():
        """
        Return latest gesture label + recent events.
        Query:
          - since_ts: float (optional)
          - limit: int (optional)
        """
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
