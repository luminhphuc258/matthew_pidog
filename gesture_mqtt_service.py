#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import socket
import threading
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

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
# CONFIG
# =========================
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "/tmp/gesture_latest.jpg")
SNAPSHOT_JPEG_QUALITY = int(os.environ.get("SNAPSHOT_JPEG_QUALITY", "80"))
SNAPSHOT_INTERVAL_SEC = float(os.environ.get("SNAPSHOT_INTERVAL_SEC", "5.0"))

CAM_DECISION_INTERVAL_SEC = float(os.environ.get("CAM_DECISION_INTERVAL_SEC", "0.30"))

GESTURE_COOLDOWN_SEC = float(os.environ.get("GESTURE_COOLDOWN_SEC", "0.35"))
GESTURE_RING_MAX = int(os.environ.get("GESTURE_RING_MAX", "60"))

ALWAYS_STOPMUSIC_ON_ANY_GESTURE = os.environ.get("ALWAYS_STOPMUSIC_ON_ANY_GESTURE", "1").strip() == "1"

# Rotate at source so directions match (avoid up/down left/right reversed)
CAM_ROTATE_180 = os.environ.get("CAM_ROTATE_180", "1").strip() == "1"

# ===== Near obstacle by "occlusion size" =====
# If biggest component bbox covers >= threshold of frame => obstacle yes
OBS_BBOX_AREA_RATIO_TH = float(os.environ.get("OBS_BBOX_AREA_RATIO_TH", "0.28"))  # ~28% frame area
OBS_BBOX_W_RATIO_TH    = float(os.environ.get("OBS_BBOX_W_RATIO_TH", "0.35"))     # bbox width >= 35% frame width
OBS_BBOX_H_RATIO_TH    = float(os.environ.get("OBS_BBOX_H_RATIO_TH", "0.35"))     # bbox height >= 35% frame height

# Binary mask threshold (gradient)
OBS_EDGE_PERCENTILE = float(os.environ.get("OBS_EDGE_PERCENTILE", "80"))  # keep top 20% gradients
OBS_MORPH_ITERS = int(os.environ.get("OBS_MORPH_ITERS", "2"))


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
# Shared Camera
# =========================
class Camera:
    def __init__(self, dev="/dev/video0", w=640, h=480, fps=30, rotate180: bool = True):
        self.rotate180 = bool(rotate180)
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._lock = threading.Lock()
        self.last = None
        self.ts = 0.0
        self._last_snapshot_ts = 0.0

        try:
            os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
        except Exception:
            pass

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None

        if self.rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        now = time.time()
        with self._lock:
            self.last = frame
            self.ts = now

            if (now - self._last_snapshot_ts) >= SNAPSHOT_INTERVAL_SEC:
                self._last_snapshot_ts = now
                _write_snapshot_atomic(self.last, SNAPSHOT_PATH, quality=SNAPSHOT_JPEG_QUALITY)

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
# CameraObstacleDetector: near obstacle by biggest "occlusion"
# - works with partial shape (bottle close, pillow close, etc.)
# ============================================================
class CameraObstacleDetector:
    def __init__(self, interval_sec: float = 0.30):
        self.interval = float(interval_sec)
        self._lock = threading.Lock()
        self._last_run = 0.0

        self._label = "no obstacle"
        self._zone = "NONE"   # LEFT/CENTER/RIGHT/NONE
        self._ts = 0.0
        self._bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
        self._bbox_ratio = 0.0

    def compute(self, frame_bgr: np.ndarray):
        now = time.time()
        if (now - self._last_run) < self.interval:
            return
        self._last_run = now

        h, w = frame_bgr.shape[:2]
        frame_area = float(w * h)

        # downscale for stable + faster
        small_w = 200
        small_h = int(h * (small_w / float(w)))
        small = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # gradient magnitude (Sobel)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        # dynamic threshold by percentile
        thr = np.percentile(mag, OBS_EDGE_PERCENTILE)
        mask = (mag >= thr).astype(np.uint8) * 255

        # close/dilate to form solid blobs
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        mask = cv2.dilate(mask, None, iterations=max(1, OBS_MORPH_ITERS))

        # connected components / contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            with self._lock:
                self._label = "no obstacle"
                self._zone = "NONE"
                self._ts = now
                self._bbox = None
                self._bbox_ratio = 0.0
            return

        # pick largest bbox
        best = None
        best_area = 0.0
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            a = float(bw * bh)
            if a > best_area:
                best_area = a
                best = (x, y, bw, bh)

        # scale bbox back to original
        sx = w / float(small_w)
        sy = h / float(small_h)
        x, y, bw, bh = best
        X = int(x * sx); Y = int(y * sy)
        BW = int(bw * sx); BH = int(bh * sy)

        bbox_area = float(BW * BH)
        bbox_ratio = bbox_area / (frame_area + 1e-9)
        bw_ratio = BW / float(w + 1e-9)
        bh_ratio = BH / float(h + 1e-9)

        # Determine obstacle yes/no: big occlusion
        obstacle_yes = (
            bbox_ratio >= OBS_BBOX_AREA_RATIO_TH
            and (bw_ratio >= OBS_BBOX_W_RATIO_TH or bh_ratio >= OBS_BBOX_H_RATIO_TH)
        )

        # zone by bbox center x
        cx = X + BW // 2
        if obstacle_yes:
            if cx < int(w * 0.33):
                zone = "LEFT"
            elif cx > int(w * 0.66):
                zone = "RIGHT"
            else:
                zone = "CENTER"
            label = "yes have obstacle"
        else:
            zone = "NONE"
            label = "no obstacle"

        with self._lock:
            self._label = label
            self._zone = zone
            self._ts = now
            self._bbox = (X, Y, BW, BH) if obstacle_yes else None
            self._bbox_ratio = bbox_ratio

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "label": self._label,
                "zone": self._zone,
                "ts": self._ts,
                "bbox": ({"x": self._bbox[0], "y": self._bbox[1], "w": self._bbox[2], "h": self._bbox[3]}
                         if self._bbox else None),
                "bbox_ratio": self._bbox_ratio,
            }

    def draw_overlay(self, frame_bgr: np.ndarray, gesture_label: str) -> np.ndarray:
        st = self.get_state()
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        obstacle_yes = (st["label"] == "yes have obstacle")
        zone = st.get("zone", "NONE")

        # draw bbox if yes
        bb = st.get("bbox")
        if bb:
            x, y, bw, bh = bb["x"], bb["y"], bb["w"], bb["h"]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        # top info bar (gesture only to avoid messy)
        bar_h = 42
        cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.putText(out, f"GESTURE: {gesture_label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # right-side obstacle box
        box_w = 260
        box_h = 80
        x0 = w - box_w - 10
        y0 = bar_h + 10
        x1 = w - 10
        y1 = y0 + box_h

        # background color: green=no, red=yes
        if obstacle_yes:
            bg = (0, 0, 120)
            txt = (255, 255, 255)
        else:
            bg = (0, 120, 0)
            txt = (255, 255, 255)

        cv2.rectangle(out, (x0, y0), (x1, y1), bg, -1)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)

        obs_txt = "YES" if obstacle_yes else "NO"
        cv2.putText(out, f"OBSTACLE: {obs_txt}", (x0 + 12, y0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, txt, 2)
        cv2.putText(out, f"ZONE: {zone}", (x0 + 12, y0 + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, txt, 2)

        return out


# =========================
# Gesture internal store
# =========================
gesture_lock = threading.Lock()
gesture_latest: Dict[str, Any] = {"label": "NA", "ts": 0.0, "face": "", "raw": ""}
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
# Attach endpoints to dashboard Flask (port 8000)
# =========================
def attach_routes_to_flask_app(flask_app: Flask, detector: CameraObstacleDetector):
    @flask_app.get("/take_gesture_meaning")
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

    @flask_app.get("/take_camera_decision")
    def take_camera_decision():
        return jsonify({"ok": True, **detector.get_state()})

    @flask_app.get("/api/gesture_status")
    def api_gesture_status():
        st = detector.get_state()
        with gesture_lock:
            gl = dict(gesture_latest)
            ring_n = len(gesture_ring)
        return jsonify({
            "ok": True,
            "snapshot_path": SNAPSHOT_PATH,
            "camera_decision": st,
            "gesture_latest": gl,
            "gesture_ring_n": ring_n,
            "cam_rotate_180": CAM_ROTATE_180,
        })


# =========================
# Main
# =========================
def main():
    cam = Camera(dev="/dev/video0", w=640, h=480, fps=30, rotate180=CAM_ROTATE_180)
    detector = CameraObstacleDetector(interval_sec=CAM_DECISION_INTERVAL_SEC)

    # Raw frame for HandCommand (already rotated correctly)
    def get_frame_raw():
        return cam.get_frame()

    # Dashboard overlay frame
    def get_frame_for_dashboard():
        frame = cam.get_frame()
        if frame is None:
            return None

        detector.compute(frame)

        with gesture_lock:
            g_lbl = gesture_latest.get("label", "NA")

        return detector.draw_overlay(frame, gesture_label=g_lbl)

    # ✅ FIX: swap STANDUP <-> SIT
    def swap_sit_stand(a: str) -> str:
        if a == "STANDUP":
            return "SIT"
        if a == "SIT":
            return "STANDUP"
        return a

    def on_action(action: str, face: str, bark: bool):
        a = (action or "").strip().upper()
        f = (face or "").strip()
        if not a:
            return

        a = swap_sit_stand(a)

        if f:
            set_face(f)

        _push_gesture(a, face=f, raw=a)

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
        get_frame_bgr=get_frame_for_dashboard,
        avoid_obstacle=None,
        on_manual_cmd=lambda m: print("[MANUAL CMD]", m, flush=True),
        rotate180=False,   # already rotated in Camera
        mqtt_enable=False,
        hand_command=hc,
    )

    attached = False
    try:
        if hasattr(dash, "app") and isinstance(dash.app, Flask):
            attach_routes_to_flask_app(dash.app, detector)
            attached = True
    except Exception as e:
        print("[WARN] cannot attach routes into WebDashboard app:", e, flush=True)

    print("\n=== Gesture Service (NO MQTT) + WebDashboard ===", flush=True)
    print("WebDashboard: http://<pi_ip>:8000", flush=True)
    if attached:
        print("Endpoints: /take_gesture_meaning  /take_camera_decision  /api/gesture_status", flush=True)
    print("Snapshot:", SNAPSHOT_PATH, flush=True)

    try:
        dash.run()
    finally:
        hc.stop()
        cam.close()


if __name__ == "__main__":
    main()
