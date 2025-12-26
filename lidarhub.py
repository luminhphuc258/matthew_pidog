#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import math
import json
import threading
import subprocess
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, jsonify, Response, request

# =======================
# CONFIG (env overridable)
# =======================
RPLIDAR_BIN = os.environ.get(
    "RPLIDAR_BIN",
    os.path.expanduser("~/rplidar_sdk/output/Linux/Release/ultra_simple")
)
RPLIDAR_PORT = os.environ.get("RPLIDAR_PORT", "/dev/ttyUSB0")
RPLIDAR_BAUD = int(os.environ.get("RPLIDAR_BAUD", "460800"))

HTTP_HOST = os.environ.get("HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("PORT", "9399"))

MAX_RANGE_M = float(os.environ.get("MAX_RANGE_M", "12.0"))
POINT_LIMIT = int(os.environ.get("POINT_LIMIT", "2500"))

# Throttle endpoints to save energy (300ms default)
THROTTLE_MS = int(os.environ.get("THROTTLE_MS", "300"))
THROTTLE_S = THROTTLE_MS / 1000.0

# Decision / Prediction settings
STOP_NEAR_M = float(os.environ.get("STOP_NEAR_M", "0.30"))      # stop if obstacle very near
PREDICT_T_SEC = float(os.environ.get("PREDICT_T_SEC", "1.0"))   # predict collision within next 1s
ROBOT_SPEED_MPS = float(os.environ.get("ROBOT_SPEED_MPS", "0.35"))  # estimated forward speed (tune!)
SAFETY_MARGIN_M = float(os.environ.get("SAFETY_MARGIN_M", "0.10"))

# Angle calibration
# FRONT_CENTER_DEG defines forward direction in LiDAR frame.
FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))

# Sector widths
FRONT_WIDTH_DEG = float(os.environ.get("FRONT_WIDTH_DEG", "30.0"))   # narrow corridor
WIDE_WIDTH_DEG  = float(os.environ.get("WIDE_WIDTH_DEG", "70.0"))    # wider for left/right/back

# Ultra_simple output sample:
# theta: 353.17 Dist: 02277.00 Q: 47
LINE_RE = re.compile(r"theta:\s*([0-9.]+)\s+Dist:\s*([0-9.]+)\s+Q:\s*(\d+)")

# =======================
# APP + STATE
# =======================
app = Flask(__name__)

lock = threading.Lock()

# Each point: (theta_deg, dist_m, q, x_m, y_m, ts)
latest_points: List[Tuple[float, float, int, float, float, float]] = []
latest_ts: float = 0.0

status: Dict[str, Any] = {
    "running": False,
    "last_error": "",
    "port": RPLIDAR_PORT,
    "baud": RPLIDAR_BAUD,
    "bin": RPLIDAR_BIN,
}

proc: Optional[subprocess.Popen] = None
stop_flag = False

# Cached payloads for throttling
cache_lock = threading.Lock()
_points_cache: Dict[str, Any] = {"ts": 0.0, "payload": None}
_decision_cache: Dict[str, Any] = {"ts": 0.0, "payload": None}

# Latest "published locally" decision label (for other classes to pull)
decision_state_lock = threading.Lock()
latest_decision_label: str = "STOP"
latest_decision_full: Dict[str, Any] = {"ok": False, "label": "STOP", "reason": "init"}


# =======================
# Helpers
# =======================
def _wrap_deg(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a


def _ang_dist_deg(a: float, b: float) -> float:
    """Smallest absolute difference between angles (deg)."""
    d = abs(_wrap_deg(a) - _wrap_deg(b))
    return min(d, 360.0 - d)


def polar_to_xy_m(theta_deg: float, dist_m: float) -> Tuple[float, float]:
    th = math.radians(theta_deg)
    x = dist_m * math.cos(th)
    y = dist_m * math.sin(th)
    return x, y


def _spawn_ultra_simple() -> subprocess.Popen:
    cmd = [
        "stdbuf", "-oL", "-eL",
        RPLIDAR_BIN,
        "--channel", "--serial",
        RPLIDAR_PORT,
        str(RPLIDAR_BAUD),
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )


def _reader_loop():
    global proc, latest_ts, stop_flag

    stop_flag = False
    status["last_error"] = ""

    try:
        proc = _spawn_ultra_simple()
    except Exception as e:
        status["running"] = False
        status["last_error"] = f"Cannot start ultra_simple: {e}"
        return

    if not proc.stdout:
        status["running"] = False
        status["last_error"] = "ultra_simple process has no stdout"
        return

    status["running"] = True

    try:
        for line in proc.stdout:
            if stop_flag:
                break

            m = LINE_RE.search(line)
            if not m:
                continue

            theta = float(m.group(1))
            dist_mm = float(m.group(2))
            q = int(m.group(3))

            if dist_mm <= 1:
                continue

            dist_m = dist_mm / 1000.0
            if dist_m > MAX_RANGE_M:
                continue

            ts = time.time()
            x_m, y_m = polar_to_xy_m(theta, dist_m)

            with lock:
                latest_points.append((theta, dist_m, q, x_m, y_m, ts))
                if len(latest_points) > POINT_LIMIT:
                    latest_points[:] = latest_points[-POINT_LIMIT:]
                latest_ts = ts

    except Exception as e:
        status["last_error"] = f"Reader loop error: {e}"
    finally:
        status["running"] = False
        try:
            if proc and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def lidar_thread_main():
    """Auto-restart ultra_simple if it exits."""
    while True:
        _reader_loop()

        if stop_flag:
            return

        # crash -> wait then restart
        time.sleep(1.0)


def _min_dist_in_sector(
    points: List[Tuple[float, float, int, float, float, float]],
    center_deg: float,
    width_deg: float,
    recent_sec: float = 0.7
) -> float:
    """
    Return minimum distance (m) among recent points within a sector.
    If none found, return +inf.
    """
    now = time.time()
    half = width_deg / 2.0
    best = float("inf")

    for (theta, dist_m, q, x, y, ts) in points:
        if now - ts > recent_sec:
            continue
        if _ang_dist_deg(theta, center_deg) <= half:
            if dist_m < best:
                best = dist_m

    return best


def _compute_decision_snapshot() -> Dict[str, Any]:
    """
    Decide safe corridor direction.
    Output label: GO_STRAIGHT / TURN_LEFT / TURN_RIGHT / STOP / GO_BACK
    """
    with lock:
        pts = list(latest_points)
        last_ts = latest_ts

    if not pts or (time.time() - last_ts) > 2.0:
        return {
            "ok": False,
            "label": "STOP",
            "reason": "no_recent_lidar",
            "ts": time.time(),
        }

    front = _wrap_deg(FRONT_CENTER_DEG)
    left  = _wrap_deg(front + 90.0)
    right = _wrap_deg(front - 90.0)
    back  = _wrap_deg(front + 180.0)

    diag_l = _wrap_deg(front + 45.0)
    diag_r = _wrap_deg(front - 45.0)

    # distances
    d_front_narrow = _min_dist_in_sector(pts, front, FRONT_WIDTH_DEG)
    d_front_wide   = _min_dist_in_sector(pts, front, WIDE_WIDTH_DEG)
    d_left_wide    = _min_dist_in_sector(pts, left,  WIDE_WIDTH_DEG)
    d_right_wide   = _min_dist_in_sector(pts, right, WIDE_WIDTH_DEG)
    d_back_wide    = _min_dist_in_sector(pts, back,  WIDE_WIDTH_DEG)

    d_diag_l = _min_dist_in_sector(pts, diag_l, 35.0)
    d_diag_r = _min_dist_in_sector(pts, diag_r, 35.0)

    # Predict collision threshold (within next 1s)
    predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
    danger_front = d_front_narrow <= max(STOP_NEAR_M, predict_dist)

    # HARD STOP if very near
    if d_front_narrow <= STOP_NEAR_M:
        return {
            "ok": True,
            "label": "STOP",
            "reason": "obstacle_very_near",
            "ts": time.time(),
            "dist": {
                "front_narrow": d_front_narrow,
                "front_wide": d_front_wide,
                "left": d_left_wide,
                "right": d_right_wide,
                "back": d_back_wide,
                "diag_left": d_diag_l,
                "diag_right": d_diag_r,
            },
            "threshold": {
                "stop_near_m": STOP_NEAR_M,
                "predict_dist_m": predict_dist,
                "predict_t_sec": PREDICT_T_SEC,
                "robot_speed_mps": ROBOT_SPEED_MPS,
                "safety_margin_m": SAFETY_MARGIN_M,
            }
        }

    # If predicted danger: turn early
    if danger_front:
        side_block = max(STOP_NEAR_M, predict_dist)

        left_ok  = max(d_left_wide, d_diag_l) > side_block
        right_ok = max(d_right_wide, d_diag_r) > side_block

        if not left_ok and not right_ok:
            if d_back_wide <= side_block:
                label = "STOP"
                reason = "boxed_in"
            else:
                label = "GO_BACK"
                reason = "front_and_sides_blocked"
        else:
            score_left = min(d_left_wide, d_diag_l)
            score_right = min(d_right_wide, d_diag_r)

            if score_left > score_right:
                label = "TURN_LEFT"
                reason = "front_predicted_collision_choose_left"
            else:
                label = "TURN_RIGHT"
                reason = "front_predicted_collision_choose_right"

        return {
            "ok": True,
            "label": label,
            "reason": reason,
            "ts": time.time(),
            "dist": {
                "front_narrow": d_front_narrow,
                "front_wide": d_front_wide,
                "left": d_left_wide,
                "right": d_right_wide,
                "back": d_back_wide,
                "diag_left": d_diag_l,
                "diag_right": d_diag_r,
            },
            "threshold": {
                "stop_near_m": STOP_NEAR_M,
                "predict_dist_m": predict_dist,
                "predict_t_sec": PREDICT_T_SEC,
                "robot_speed_mps": ROBOT_SPEED_MPS,
                "safety_margin_m": SAFETY_MARGIN_M,
            }
        }

    # Otherwise safe
    return {
        "ok": True,
        "label": "GO_STRAIGHT",
        "reason": "front_clear",
        "ts": time.time(),
        "dist": {
            "front_narrow": d_front_narrow,
            "front_wide": d_front_wide,
            "left": d_left_wide,
            "right": d_right_wide,
            "back": d_back_wide,
            "diag_left": d_diag_l,
            "diag_right": d_diag_r,
        },
        "threshold": {
            "stop_near_m": STOP_NEAR_M,
            "predict_dist_m": predict_dist,
            "predict_t_sec": PREDICT_T_SEC,
            "robot_speed_mps": ROBOT_SPEED_MPS,
            "safety_margin_m": SAFETY_MARGIN_M,
        }
    }


def _build_points_payload(limit: int = 1600) -> Dict[str, Any]:
    with lock:
        pts = list(latest_points[-limit:])
        ts = latest_ts

    out = []
    for (theta, dist_m, q, x, y, t) in pts:
        out.append({
            "theta": theta,
            "dist_m": dist_m,
            "q": q,
            "x": x,
            "y": y,
            "ts": t,
        })

    return {
        "ok": True,
        "ts": time.time(),
        "last_point_ts": ts,
        "n": len(out),
        "points": out,
        "frame": {
            "front_center_deg": FRONT_CENTER_DEG,
            "max_range_m": MAX_RANGE_M,
        }
    }


# =======================
# Background Workers (throttle 300ms)
# =======================
def decision_worker():
    global latest_decision_label, latest_decision_full

    while True:
        payload = _compute_decision_snapshot()

        with decision_state_lock:
            latest_decision_full = payload
            latest_decision_label = payload.get("label", "STOP")

        with cache_lock:
            _decision_cache["ts"] = time.time()
            _decision_cache["payload"] = payload

        time.sleep(THROTTLE_S)


def points_worker():
    while True:
        payload = _build_points_payload()

        with cache_lock:
            _points_cache["ts"] = time.time()
            _points_cache["payload"] = payload

        time.sleep(THROTTLE_S)


# =======================
# ROUTES
# =======================
@app.get("/api/status")
def api_status():
    with lock:
        pts_n = len(latest_points)
        ts = latest_ts
    with decision_state_lock:
        lbl = latest_decision_label

    return jsonify({
        "running": status["running"],
        "port": status["port"],
        "baud": status["baud"],
        "bin": status["bin"],
        "points_buffered": pts_n,
        "last_point_ts": ts,
        "age_s": (time.time() - ts) if ts else None,
        "last_error": status["last_error"],
        "pid": proc.pid if proc else None,
        "throttle_ms": THROTTLE_MS,
        "latest_label": lbl,
    })


@app.get("/take_lidar_data")
def take_lidar_data():
    """
    Main endpoint for WebDashboard (3D giả).
    Throttled by background cache (~300ms).
    """
    with cache_lock:
        payload = _points_cache["payload"]

    if payload is None:
        payload = _build_points_payload()
        with cache_lock:
            _points_cache["payload"] = payload
            _points_cache["ts"] = time.time()

    return jsonify(payload)


@app.get("/ask_lidar_decision")
def ask_lidar_decision():
    """
    Important endpoint: returns latest decision.
    Throttled by background cache (~300ms).
    """
    with cache_lock:
        payload = _decision_cache["payload"]

    if payload is None:
        payload = _compute_decision_snapshot()
        with cache_lock:
            _decision_cache["payload"] = payload
            _decision_cache["ts"] = time.time()

    return jsonify(payload)


@app.get("/api/decision_label")
def api_decision_label():
    """
    For other classes (main/motion controller) that only need the label.
    Returns plain text.
    """
    with decision_state_lock:
        lbl = latest_decision_label
    return Response(lbl, mimetype="text/plain")


@app.get("/api/decision")
def api_decision():
    """
    Full decision JSON (same as ask_lidar_decision, but stable for internal pull).
    """
    with decision_state_lock:
        payload = dict(latest_decision_full)
    return jsonify(payload)


@app.post("/api/restart")
def api_restart():
    """
    Restart ultra_simple process.
    """
    global stop_flag, proc
    stop_flag = True
    try:
        if proc and proc.poll() is None:
            proc.terminate()
    except Exception:
        pass

    time.sleep(0.4)

    stop_flag = False
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    return jsonify({"ok": True})


@app.get("/")
def home():
    return Response(
        "<h3>lidarhub running</h3>"
        "<ul>"
        "<li>/take_lidar_data</li>"
        "<li>/ask_lidar_decision</li>"
        "<li>/api/decision_label</li>"
        "<li>/api/status</li>"
        "<li>/api/restart (POST)</li>"
        "</ul>",
        mimetype="text/html"
    )


def main():
    # Start background threads
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    threading.Thread(target=points_worker, daemon=True).start()
    threading.Thread(target=decision_worker, daemon=True).start()

    # Run Flask
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
