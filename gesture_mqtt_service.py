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
from flask import Flask, jsonify, request

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
# Local API config
# =========================
API_HOST = os.environ.get("GESTURE_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("GESTURE_API_PORT", "8001"))

# Snapshot export
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
# - Find vertical stripe region whose pixels are uniform (low std)
#   and stable over time (2 seconds window) + low edges
# - Stripe height must be close to frame height (ROI)
# - Stripe width can be variable
# - Decide zone by stripe center X: LEFT/CENTER/RIGHT
# ============================================================
class ColorStripeObstacleDetector:
    def __init__(self,
                 decision_interval_sec: float = 0.10,
                 window_sec: float = 2.0,
                 roi_y1_ratio: float = 0.10,
                 roi_y2_ratio: float = 0.96,
                 min_stripe_width_ratio: float = 0.08,   # >= 8% frame width
                 max_stripe_width_ratio: float = 0.70,   # avoid “nearly whole frame”
                 # thresholds for "uniform"
                 col_std_v_thr: float = 18.0,            # per-column std(V)
                 col_edge_thr: float = 0.040,            # per-column edge density
                 # decide stable within 2 seconds
                 stable_mean_delta_thr: float = 10.0,    # mean(V) drift allowed
                 min_good_ratio_in_window: float = 0.65  # fraction of frames in window to confirm
                 ):
        self.interval = float(decision_interval_sec)
        self.window_sec = float(window_sec)

        self.ry1 = float(roi_y1_ratio)
        self.ry2 = float(roi_y2_ratio)

        self.min_w_ratio = float(min_stripe_width_ratio)
        self.max_w_ratio = float(max_stripe_width_ratio)

        self.col_std_v_thr = float(col_std_v_thr)
        self.col_edge_thr = float(col_edge_thr)
        self.stable_mean_delta_thr = float(stable_mean_delta_thr)
        self.min_good_ratio_in_window = float(min_good_ratio_in_window)

        self._lock = threading.Lock()
        self._last_run = 0.0

        # rolling window history: (ts, ok, x1, x2, meanV)
        self._hist: deque = deque(maxlen=200)

        self._label = "no obstacle"
        self._zone = "NONE"
        self._ts = 0.0
        self._stripe: Optional[Tuple[int, int, int, int]] = None  # x1,x2,y1,y2

    @staticmethod
    def _zone_from_x(cx: int, w: int) -> str:
        if cx < w * (1/3):
            return "LEFT"
        if cx > w * (2/3):
            return "RIGHT"
        return "CENTER"

    def _find_best_stripe(self, roi_bgr: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        """
        Return:
          ok, (x1,x2) in ROI coordinates, meanV of stripe
        """
        h, w = roi_bgr.shape[:2]
        if w < 50 or h < 50:
            return False, None, 0.0

        small_w = 320
        scale = small_w / float(w) if w > small_w else 1.0
        roi = cv2.resize(roi_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else roi_bgr
        hh, ww = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edges = (edges > 0).astype(np.float32)

        # per-column stats
        col_std_v = np.std(v, axis=0)  # (ww,)
        col_edge = np.mean(edges, axis=0)  # (ww,)

        good = (col_std_v < self.col_std_v_thr) & (col_edge < self.col_edge_thr)

        # find longest contiguous run
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
                best_run = (i, j)  # [i, j)
            i = j

        if best_run is None:
            return False, None, 0.0

        min_w = int(self.min_w_ratio * ww)
        max_w = int(self.max_w_ratio * ww)

        if best_len < max(10, min_w) or best_len > max_w:
            return False, None, 0.0

        x1s, x2s = best_run
        stripe_v = v[:, x1s:x2s]
        meanV = float(np.mean(stripe_v))

        # map back to original ROI coords
        if scale != 1.0:
            x1 = int(x1s / scale)
            x2 = int(x2s / scale)
        else:
            x1, x2 = int(x1s), int(x2s)

        # clamp
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w, x2))
        if x2 - x1 < int(self.min_w_ratio * w):
            return False, None, 0.0

        return True, (x1, x2), meanV

    def compute(self, frame_bgr: np.ndarray):
        now = time.time()
        if (now - self._last_run) < self.interval:
            return
        self._last_run = now

        H, W = frame_bgr.shape[:2]
        y1 = int(H * self.ry1)
        y2 = int(H * self.ry2)
        y1 = max(0, min(H-1, y1))
        y2 = max(y1+10, min(H, y2))

        roi = frame_bgr[y1:y2, :]

        ok, run, meanV = self._find_best_stripe(roi)
        x1 = x2 = -1
        if ok and run is not None:
            x1, x2 = run[0], run[1]

        # push history
        self._hist.append((now, bool(ok), int(x1), int(x2), float(meanV)))

        # evaluate window 2s
        tmin = now - self.window_sec
        items = [it for it in list(self._hist) if it[0] >= tmin]
        if len(items) < 6:
            # not enough evidence
            with self._lock:
                self._label = "no obstacle"
                self._zone = "NONE"
                self._ts = now
                self._stripe = None
            return

        oks = [it for it in items if it[1]]
        good_ratio = len(oks) / float(len(items))

        if good_ratio < self.min_good_ratio_in_window:
            with self._lock:
                self._label = "no obstacle"
                self._zone = "NONE"
                self._ts = now
                self._stripe = None
            return

        # stability: meanV drift in window must be small
        vs = [it[4] for it in oks]
        if not vs:
            with self._lock:
                self._label = "no obstacle"
                self._zone = "NONE"
                self._ts = now
                self._stripe = None
            return

        if (max(vs) - min(vs)) > self.stable_mean_delta_thr:
            with self._lock:
                self._label = "no obstacle"
                self._zone = "NONE"
                self._ts = now
                self._stripe = None
            return

        # choose median stripe from oks
        oks_sorted = sorted(oks, key=lambda x: (x[3]-x[2]), reverse=True)
        best = oks_sorted[0]
        _, _, bx1, bx2, _ = best

        cx = int((bx1 + bx2) / 2)
        zone = self._zone_from_x(cx, W)

        with self._lock:
            self._label = "yes have obstacle"
            self._zone = zone
            self._ts = now
            self._stripe = (int(bx1), int(bx2), int(y1), int(y2))

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            stripe = self._stripe
            return {
                "label": self._label,
                "zone": self._zone,
                "ts": self._ts,
                "stripe": None if stripe is None else {"x1": stripe[0], "x2": stripe[1], "y1": stripe[2], "y2": stripe[3]},
            }

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        st = self.get_state()
        out = frame_bgr.copy()

        # draw stripe only if obstacle YES
        if st.get("label") == "yes have obstacle" and st.get("stripe"):
            s = st["stripe"]
            x1, x2, y1, y2 = int(s["x1"]), int(s["x2"]), int(s["y1"]), int(s["y2"])
            x1 = max(0, min(out.shape[1]-1, x1))
            x2 = max(0, min(out.shape[1], x2))
            if x2 > x1:
                overlay = out.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                out = cv2.addWeighted(overlay, 0.20, out, 0.80, 0)

                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return out


# =========================
# Gesture “local publish” store
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
# Local API (Flask)
# =========================
api_app = Flask("gesture_local_api")
_detector: Optional[ColorStripeObstacleDetector] = None


@api_app.get("/take_gesture_meaning")
def take_gesture_meaning():
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


@api_app.get("/take_camera_decision")
def take_camera_decision():
    if _detector is None:
        return jsonify({"ok": False, "error": "detector_not_ready"}), 503
    st = _detector.get_state()
    return jsonify({"ok": True, **st})


def run_api_server():
    api_app.run(host=API_HOST, port=API_PORT, debug=False, threaded=True)


# =========================
# Main
# =========================
def main():
    global _detector

    cam = Camera(dev="/dev/video0", w=640, h=480, fps=30,
                 snapshot_path=SNAPSHOT_PATH,
                 snapshot_interval_sec=SNAPSHOT_INTERVAL_SEC,
                 snapshot_quality=SNAPSHOT_JPEG_QUALITY)

    _detector = ColorStripeObstacleDetector(
        decision_interval_sec=0.10,
        window_sec=2.0,
        roi_y1_ratio=0.10,
        roi_y2_ratio=0.96,
        min_stripe_width_ratio=0.08,
        max_stripe_width_ratio=0.70,
        col_std_v_thr=18.0,
        col_edge_thr=0.040,
        stable_mean_delta_thr=10.0,
        min_good_ratio_in_window=0.65
    )

    # Start local API server (8001)
    threading.Thread(target=run_api_server, daemon=True).start()

    # Dashboard frame: overlay obstacle stripe ONLY (no big text on image)
    def get_frame_for_dashboard():
        frame = cam.get_frame()
        if frame is None:
            return None
        _detector.compute(frame)
        return _detector.draw_overlay(frame)

    def get_frame_raw():
        return cam.get_frame()

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

    # Pass obstacle getter into dashboard so Quick Status can show it
    dash = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_frame_bgr=get_frame_for_dashboard,
        get_obstacle_state=lambda: _detector.get_state() if _detector else {"label": "no obstacle", "zone": "NONE"},
        on_manual_cmd=lambda m: print("[MANUAL CMD]", m, flush=True),
        rotate180=True,
        hand_command=hc,
    )

    print("\n=== Gesture Service (NO MQTT) + WebDashboard ===", flush=True)
    print("WebDashboard: http://<pi_ip>:8000", flush=True)
    print("Gesture API :", f"http://<pi_ip>:{API_PORT}/take_gesture_meaning", flush=True)
    print("Camera API  :", f"http://<pi_ip>:{API_PORT}/take_camera_decision", flush=True)
    print("Snapshot    :", SNAPSHOT_PATH, flush=True)

    try:
        dash.run()
    finally:
        hc.stop()
        cam.close()


if __name__ == "__main__":
    main()
