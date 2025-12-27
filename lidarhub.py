#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import math
import threading
import subprocess
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, jsonify, Response

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

# Endpoint caches (ms)
THROTTLE_MS = int(os.environ.get("THROTTLE_MS", "250"))
THROTTLE_S = THROTTLE_MS / 1000.0

# Decision thresholds (meters)
STOP_NEAR_M = float(os.environ.get("STOP_NEAR_M", "0.30"))           # "too close" band
PREDICT_T_SEC = float(os.environ.get("PREDICT_T_SEC", "1.0"))
ROBOT_SPEED_MPS = float(os.environ.get("ROBOT_SPEED_MPS", "0.35"))
SAFETY_MARGIN_M = float(os.environ.get("SAFETY_MARGIN_M", "0.10"))

# Angle calibration
FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))

# Sector widths (deg)
FRONT_WIDTH_DEG = float(os.environ.get("FRONT_WIDTH_DEG", "30.0"))
WIDE_WIDTH_DEG  = float(os.environ.get("WIDE_WIDTH_DEG", "70.0"))

# Sticky decision to avoid "STOP then stuck"
TURN_STICKY_SEC = float(os.environ.get("TURN_STICKY_SEC", "1.2"))   # keep TURN for this long once chosen
BACK_STICKY_SEC = float(os.environ.get("BACK_STICKY_SEC", "0.9"))

# ultra_simple output sample:
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

# Cached payloads
cache_lock = threading.Lock()
_points_cache: Dict[str, Any] = {"ts": 0.0, "payload": None}
_decision_cache: Dict[str, Any] = {"ts": 0.0, "payload": None}

# Exposed decision state
decision_state_lock = threading.Lock()
latest_decision_label: str = "STOP"
latest_decision_full: Dict[str, Any] = {"ok": False, "label": "STOP", "reason": "init", "ts": 0.0}

# Sticky turn/back behavior
sticky_lock = threading.Lock()
sticky_label: str = ""
sticky_until_ts: float = 0.0

# =======================
# Helpers
# =======================
def _wrap_deg(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def _ang_dist_deg(a: float, b: float) -> float:
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
    while True:
        _reader_loop()
        if stop_flag:
            return
        time.sleep(1.0)

def _min_dist_in_sector(
    points: List[Tuple[float, float, int, float, float, float]],
    center_deg: float,
    width_deg: float,
    recent_sec: float = 0.7
) -> float:
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

def _set_sticky(label: str, sec: float):
    global sticky_label, sticky_until_ts
    with sticky_lock:
        sticky_label = label
        sticky_until_ts = time.time() + float(sec)

def _get_sticky() -> Optional[str]:
    with sticky_lock:
        if sticky_label and time.time() < sticky_until_ts:
            return sticky_label
    return None

def _clear_sticky():
    global sticky_label, sticky_until_ts
    with sticky_lock:
        sticky_label = ""
        sticky_until_ts = 0.0

def _compute_decision_snapshot() -> Dict[str, Any]:
    """
    Output label:
      - GO_STRAIGHT
      - TURN_LEFT
      - TURN_RIGHT
      - GO_BACK
      - STOP
    Key behavior:
      - If front blocked: pick TURN/GO_BACK (STOP only when boxed_in)
      - Sticky TURN/BACK for a short time to allow robot to rotate/move back.
    """
    with lock:
        pts = list(latest_points)
        last_ts = latest_ts

    now = time.time()
    if not pts or (now - last_ts) > 2.0:
        _clear_sticky()
        return {
            "ok": False,
            "label": "STOP",
            "reason": "no_recent_lidar",
            "ts": now,
        }

    # If sticky decision active, return it (but still publish distances for debug)
    sticky = _get_sticky()

    front = _wrap_deg(FRONT_CENTER_DEG)
    left  = _wrap_deg(front + 90.0)
    right = _wrap_deg(front - 90.0)
    back  = _wrap_deg(front + 180.0)

    diag_l = _wrap_deg(front + 45.0)
    diag_r = _wrap_deg(front - 45.0)

    d_front_narrow = _min_dist_in_sector(pts, front, FRONT_WIDTH_DEG)
    d_front_wide   = _min_dist_in_sector(pts, front, WIDE_WIDTH_DEG)
    d_left_wide    = _min_dist_in_sector(pts, left,  WIDE_WIDTH_DEG)
    d_right_wide   = _min_dist_in_sector(pts, right, WIDE_WIDTH_DEG)
    d_back_wide    = _min_dist_in_sector(pts, back,  WIDE_WIDTH_DEG)
    d_diag_l       = _min_dist_in_sector(pts, diag_l, 35.0)
    d_diag_r       = _min_dist_in_sector(pts, diag_r, 35.0)

    predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
    side_block = max(STOP_NEAR_M, predict_dist)

    dist_pack = {
        "front_narrow": d_front_narrow,
        "front_wide": d_front_wide,
        "left": d_left_wide,
        "right": d_right_wide,
        "back": d_back_wide,
        "diag_left": d_diag_l,
        "diag_right": d_diag_r,
    }

    thresh_pack = {
        "stop_near_m": STOP_NEAR_M,
        "predict_dist_m": predict_dist,
        "predict_t_sec": PREDICT_T_SEC,
        "robot_speed_mps": ROBOT_SPEED_MPS,
        "safety_margin_m": SAFETY_MARGIN_M,
        "side_block_m": side_block,
        "sticky_turn_sec": TURN_STICKY_SEC,
        "sticky_back_sec": BACK_STICKY_SEC,
    }

    # if sticky exists, we still can break sticky when front is clearly safe
    if sticky:
        # Break sticky if front is clearly open
        if d_front_narrow > max(STOP_NEAR_M, predict_dist) * 1.25:
            _clear_sticky()
        else:
            return {
                "ok": True,
                "label": sticky,
                "reason": "sticky",
                "ts": now,
                "dist": dist_pack,
                "threshold": thresh_pack,
            }

    # ===== Main decision =====
    # Front danger band
    front_blocked = (d_front_narrow <= max(STOP_NEAR_M, predict_dist))

    if front_blocked:
        left_ok  = max(d_left_wide, d_diag_l) > side_block
        right_ok = max(d_right_wide, d_diag_r) > side_block
        back_ok  = d_back_wide > side_block

        # choose best side by score
        score_left = min(d_left_wide, d_diag_l)
        score_right = min(d_right_wide, d_diag_r)

        if left_ok or right_ok:
            if score_left >= score_right:
                label = "TURN_LEFT"
                reason = "front_blocked_choose_left"
            else:
                label = "TURN_RIGHT"
                reason = "front_blocked_choose_right"
            _set_sticky(label, TURN_STICKY_SEC)
        else:
            if back_ok:
                label = "GO_BACK"
                reason = "front_and_sides_blocked_go_back"
                _set_sticky(label, BACK_STICKY_SEC)
            else:
                label = "STOP"
                reason = "boxed_in"

        return {
            "ok": True,
            "label": label,
            "reason": reason,
            "ts": now,
            "dist": dist_pack,
            "threshold": thresh_pack,
        }

    # Front safe
    _clear_sticky()
    return {
        "ok": True,
        "label": "GO_STRAIGHT",
        "reason": "front_clear",
        "ts": now,
        "dist": dist_pack,
        "threshold": thresh_pack,
    }

def _build_points_payload(limit: int = 1600) -> Dict[str, Any]:
    """
    IMPORTANT: Provide BOTH schemas for compatibility:
      - theta/dist_m (original)
      - angle/dist_cm (your autopilot expects this)
    """
    with lock:
        pts = list(latest_points[-limit:])
        ts = latest_ts

    out = []
    for (theta, dist_m, q, x, y, t) in pts:
        out.append({
            # original
            "theta": theta,
            "dist_m": dist_m,

            # compatibility for your autopilot/map:
            "angle": theta,                 # keep 0..360 ok
            "dist_cm": dist_m * 100.0,

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
# Background Workers
# =======================
def decision_worker():
    global latest_decision_label, latest_decision_full
    while True:
        payload = _compute_decision_snapshot()

        with decision_state_lock:
            latest_decision_full = payload
            latest_decision_label = str(payload.get("label", "STOP") or "STOP")

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
    with decision_state_lock:
        lbl = latest_decision_label
    return Response(str(lbl), mimetype="text/plain")

@app.get("/api/decision")
def api_decision():
    with decision_state_lock:
        payload = dict(latest_decision_full)
    return jsonify(payload)

@app.post("/api/restart")
def api_restart():
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
        "<li>/api/decision</li>"
        "<li>/api/status</li>"
        "<li>/api/restart (POST)</li>"
        "</ul>",
        mimetype="text/html"
    )

def main():
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    threading.Thread(target=points_worker, daemon=True).start()
    threading.Thread(target=decision_worker, daemon=True).start()

    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
