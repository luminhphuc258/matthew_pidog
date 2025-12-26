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
from flask import Flask, jsonify, Response, request

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

# Throttle decision compute (camera obstacle) ~300ms
CAM_DECISION_INTERVAL_SEC = float(os.environ.get("CAM_DECISION_INTERVAL_SEC", "0.30"))

# Snapshot export
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "/tmp/gesture_latest.jpg")
SNAPSHOT_JPEG_QUALITY = int(os.environ.get("SNAPSHOT_JPEG_QUALITY", "80"))
SNAPSHOT_INTERVAL_SEC = float(os.environ.get("SNAPSHOT_INTERVAL_SEC", "5.0"))

# Gesture publish internal (avoid spam)
GESTURE_COOLDOWN_SEC = float(os.environ.get("GESTURE_COOLDOWN_SEC", "0.35"))
GESTURE_RING_MAX = int(os.environ.get("GESTURE_RING_MAX", "60"))

# ✅ Option: any hand action triggers stopmusic event
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
                print(f"[SNAP] wrote {self.snapshot_path}", flush=True)

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
# CAMERA OBSTACLE “ULTRASONIC-LIKE” DECISION (NEAR ONLY)
# - Detect big/near objects that cover large portion of frame
# - Draw outlines/bbox only for near obstacles
# ============================================================
class CameraObstacleDetector:
    def __init__(self,
                 decision_interval_sec: float = 0.30,
                 min_area_ratio: float = 0.08,      # bbox covers >= 8% of frame => near
                 center_region_ratio: float = 0.70  # bbox center must be in central region
                 ):
        self.interval = float(decision_interval_sec)
        self.min_area_ratio = float(min_area_ratio)
        self.center_region_ratio = float(center_region_ratio)

        self._lock = threading.Lock()
        self._last_run = 0.0
        self._label = "no obstacle"
        self._ts = 0.0
        self._near_contours: List[np.ndarray] = []
        self._near_boxes: List[Tuple[int, int, int, int, str]] = []  # x,y,w,h,shape

    @staticmethod
    def _shape_name(cnt: np.ndarray) -> str:
        peri = cv2.arcLength(cnt, True)
        if peri <= 1e-6:
            return "unknown"
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        v = len(approx)

        area = abs(cv2.contourArea(cnt))
        if area <= 1e-6:
            return "unknown"

        circularity = 4 * np.pi * area / (peri * peri + 1e-9)

        if v == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h + 1e-9)
            if 0.85 <= ar <= 1.15:
                return "square"
            return "rectangle"
        if v > 5:
            if circularity > 0.80:
                return "circle/sphere"
            return "ellipse"
        if circularity > 0.80:
            return "circle/sphere"
        return "object"

    def compute(self, frame_bgr: np.ndarray):
        now = time.time()
        if (now - self._last_run) < self.interval:
            return
        self._last_run = now

        h, w = frame_bgr.shape[:2]
        frame_area = float(w * h)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.dilate(edges, None, iterations=2)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        near_contours = []
        near_boxes = []

        cx_min = int(w * (1.0 - self.center_region_ratio) / 2.0)
        cx_max = int(w - cx_min)
        cy_min = int(h * (1.0 - self.center_region_ratio) / 2.0)
        cy_max = int(h - cy_min)

        for cnt in cnts:
            area = abs(cv2.contourArea(cnt))
            if area < 300:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            box_area = float(bw * bh)
            box_ratio = box_area / (frame_area + 1e-9)

            if box_ratio < self.min_area_ratio:
                continue

            center_x = x + bw // 2
            center_y = y + bh // 2
            if not (cx_min <= center_x <= cx_max and cy_min <= center_y <= cy_max):
                continue

            shape = self._shape_name(cnt)
            near_contours.append(cnt)
            near_boxes.append((x, y, bw, bh, shape))

        label = "yes have obstacle" if len(near_boxes) > 0 else "no obstacle"

        with self._lock:
            self._label = label
            self._ts = now
            self._near_contours = near_contours
            self._near_boxes = near_boxes

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "label": self._label,
                "ts": self._ts,
                "n": len(self._near_boxes),
                "boxes": [{"x": x, "y": y, "w": w, "h": h, "shape": s} for (x, y, w, h, s) in self._near_boxes],
            }

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        with self._lock:
            boxes = list(self._near_boxes)
            contours = list(self._near_contours)
            label = self._label

        out = frame_bgr.copy()
        if contours:
            cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
        for (x, y, bw, bh, shape) in boxes:
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(out, shape, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(out, f"camera_decision: {label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
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

    # cooldown per label
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
_detector: Optional[CameraObstacleDetector] = None


@api_app.get("/take_gesture_meaning")
def take_gesture_meaning():
    """
    Return latest gesture label + recent events.
    Query:
      - since_ts: float (optional) => return only events newer than this
      - limit: int (optional) => max events returned
    """
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

    return jsonify({
        "ok": True,
        "latest": latest,
        "events": events,
    })


@api_app.get("/take_camera_decision")
def take_camera_decision():
    if _detector is None:
        return jsonify({"ok": False, "error": "detector_not_ready"}), 503
    st = _detector.get_state()
    return jsonify({"ok": True, **st})


@api_app.get("/api/status")
def api_status():
    st = _detector.get_state() if _detector else {"label": "unknown", "n": 0, "ts": 0}
    with gesture_lock:
        gl = dict(gesture_latest)
        ring_n = len(gesture_ring)
    return jsonify({
        "ok": True,
        "api": {"host": API_HOST, "port": API_PORT},
        "snapshot_path": SNAPSHOT_PATH,
        "camera_decision": st,
        "gesture_latest": gl,
        "gesture_ring_n": ring_n,
    })


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

    _detector = CameraObstacleDetector(
        decision_interval_sec=CAM_DECISION_INTERVAL_SEC,
        min_area_ratio=0.08,
        center_region_ratio=0.70
    )

    # Start local API server
    threading.Thread(target=run_api_server, daemon=True).start()
    print(f"[API ] http://<pi_ip>:{API_PORT}/take_gesture_meaning", flush=True)
    print(f"[API ] http://<pi_ip>:{API_PORT}/take_camera_decision", flush=True)

    # Dashboard: overlay obstacles (near only)
    def get_frame_for_dashboard():
        frame = cam.get_frame()
        if frame is None:
            return None
        _detector.compute(frame)          # throttled inside
        return _detector.draw_overlay(frame)

    # HandCommand should use raw frame
    def get_frame_raw():
        return cam.get_frame()

    # ✅ On any hand gesture:
    # - publish internal label of actual gesture
    # - optionally also publish STOPMUSIC event
    def on_action(action: str, face: str, bark: bool):
        # Normalize common names if needed
        a = (action or "").strip().upper()
        f = (face or "").strip()

        if not a:
            return

        print(f"[GEST] {a}", flush=True)
        if f:
            set_face(f)

        # Publish the actual gesture label internally
        _push_gesture(a, face=f, raw=a)

        # Optional: any gesture triggers stopmusic event too
        if ALWAYS_STOPMUSIC_ON_ANY_GESTURE:
            _push_gesture("STOPMUSIC", face=f, raw=f"auto_from_{a}")

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

    dash = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_frame_bgr=get_frame_for_dashboard,   # ✅ annotated
        avoid_obstacle=None,
        on_manual_cmd=lambda m: print("[MANUAL CMD]", m, flush=True),
        rotate180=True,
        mqtt_enable=False,
        hand_command=hc,
    )

    print("\n=== Gesture Service (NO MQTT) + WebDashboard ===", flush=True)
    print("WebDashboard: http://<pi_ip>:8000", flush=True)
    print("Gesture API:", f"http://<pi_ip>:{API_PORT}/take_gesture_meaning", flush=True)
    print("Camera API :", f"http://<pi_ip>:{API_PORT}/take_camera_decision", flush=True)
    print("Snapshot   :", SNAPSHOT_PATH, flush=True)

    try:
        dash.run()
    finally:
        hc.stop()
        cam.close()


if __name__ == "__main__":
    main()
