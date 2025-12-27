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
STOP_NEAR_M = float(os.environ.get("STOP_NEAR_M", "0.30"))     # quá gần -> STOP ngay
PREDICT_T_SEC = float(os.environ.get("PREDICT_T_SEC", "1.0"))
ROBOT_SPEED_MPS = float(os.environ.get("ROBOT_SPEED_MPS", "0.35"))
SAFETY_MARGIN_M = float(os.environ.get("SAFETY_MARGIN_M", "0.10"))

# Angle calibration
FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))

# Sector widths (deg)
FRONT_WIDTH_DEG = float(os.environ.get("FRONT_WIDTH_DEG", "30.0"))
WIDE_WIDTH_DEG  = float(os.environ.get("WIDE_WIDTH_DEG", "70.0"))

# ===== Corridor / gap check =====
ROBOT_WIDTH_M = float(os.environ.get("ROBOT_WIDTH_M", "0.15"))           # robot ngang 15cm
CLEARANCE_MARGIN_M = float(os.environ.get("CLEARANCE_MARGIN_M", "0.03")) # margin mỗi bên
LOOKAHEAD_M = float(os.environ.get("LOOKAHEAD_M", "1.20"))               # xét trong 1.2m phía trước

# Candidate steering angles (relative to FRONT). Positive = turn LEFT.
CAND_STEP_DEG = float(os.environ.get("CAND_STEP_DEG", "10.0"))
CAND_MAX_DEG  = float(os.environ.get("CAND_MAX_DEG", "60.0"))

# ===== Behavior: stop-then-turn =====
STOP_HOLD_SEC = float(os.environ.get("STOP_HOLD_SEC", "0.35"))
TURN_HOLD_SEC = float(os.environ.get("TURN_HOLD_SEC", "1.10"))

# Back control (avoid backing too much)
BACK_HOLD_SEC = float(os.environ.get("BACK_HOLD_SEC", "0.45"))
BACK_COOLDOWN_SEC = float(os.environ.get("BACK_COOLDOWN_SEC", "2.5"))

# ultra_simple output:
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

# ===== STOP -> TURN chain state =====
_stop_turn_lock = threading.Lock()
_stop_turn_label: str = ""     # TURN_LEFT / TURN_RIGHT
_stop_turn_start: float = 0.0  # when to begin turning
_stop_turn_end: float = 0.0    # when to stop turning

# ===== BACK cooldown =====
_behavior_lock = threading.Lock()
_last_back_ts: float = 0.0
_back_until: float = 0.0

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

def _front_frame_xy(dist_m: float, theta_deg: float, front_center_deg: float, heading_rel_deg: float) -> Tuple[float, float]:
    """
    Convert polar point to XY in a frame where:
      - y is forward along (front_center + heading_rel)
      - x is lateral (left +, right -)
    """
    rel = _wrap_deg(theta_deg - (front_center_deg + heading_rel_deg))
    if rel > 180.0:
        rel -= 360.0
    rr = math.radians(rel)
    x = dist_m * math.sin(rr)
    y = dist_m * math.cos(rr)
    return x, y

def _corridor_clearance(
    pts: List[Tuple[float, float, int, float, float, float]],
    front_center_deg: float,
    heading_rel_deg: float,
    corridor_half_w: float,
    lookahead_m: float,
    recent_sec: float = 0.7
) -> float:
    """
    Clearance = min(y) of obstacle points that lie inside corridor rectangle:
      abs(x) <= corridor_half_w, 0 < y <= lookahead_m
    If none -> +inf
    """
    now = time.time()
    best_y = float("inf")

    for (theta, dist_m, q, x0, y0, ts) in pts:
        if now - ts > recent_sec:
            continue
        if dist_m <= 0.0:
            continue

        x, y = _front_frame_xy(dist_m, theta, front_center_deg, heading_rel_deg)
        if y <= 0:
            continue
        if y > lookahead_m:
            continue
        if abs(x) <= corridor_half_w:
            if y < best_y:
                best_y = y

    return best_y

def _pick_best_heading_by_corridor(pts: List[Tuple[float, float, int, float, float, float]]) -> Dict[str, Any]:
    corridor_half_w = (ROBOT_WIDTH_M / 2.0) + CLEARANCE_MARGIN_M
    max_deg = float(CAND_MAX_DEG)
    step = float(CAND_STEP_DEG)

    candidates: List[float] = []
    a = -max_deg
    while a <= max_deg + 1e-6:
        candidates.append(round(a, 3))
        a += step

    clear0 = _corridor_clearance(pts, FRONT_CENTER_DEG, 0.0, corridor_half_w, LOOKAHEAD_M)
    best_heading = 0.0
    best_clear = clear0
    best_score = (min(clear0, LOOKAHEAD_M) - 0.001 * abs(0.0))

    detail = []
    for h in candidates:
        c = _corridor_clearance(pts, FRONT_CENTER_DEG, h, corridor_half_w, LOOKAHEAD_M)
        c_cap = min(c, LOOKAHEAD_M)
        score = c_cap - 0.001 * abs(h)
        detail.append({"heading_deg": h, "clearance_m": c})
        if score > best_score:
            best_score = score
            best_heading = h
            best_clear = c

    return {
        "corridor_half_w_m": corridor_half_w,
        "lookahead_m": LOOKAHEAD_M,
        "best_heading_deg": float(best_heading),
        "best_clearance_m": float(best_clear),
        "clear0_m": float(clear0),
        "candidates": detail,
    }

def _arm_stop_then_turn(now: float, turn_label: str):
    """Schedule TURN after STOP_HOLD_SEC."""
    global _stop_turn_label, _stop_turn_start, _stop_turn_end
    with _stop_turn_lock:
        _stop_turn_label = turn_label
        _stop_turn_start = now + STOP_HOLD_SEC
        _stop_turn_end   = _stop_turn_start + TURN_HOLD_SEC

def _check_stop_then_turn(now: float) -> Optional[str]:
    """Return TURN label if we are in the TURN window."""
    global _stop_turn_label, _stop_turn_start, _stop_turn_end
    with _stop_turn_lock:
        if not _stop_turn_label:
            return None
        if now < _stop_turn_start:
            return None
        if now <= _stop_turn_end:
            return _stop_turn_label
        # expired -> clear
        _stop_turn_label = ""
        _stop_turn_start = 0.0
        _stop_turn_end = 0.0
        return None

def _clear_stop_then_turn():
    global _stop_turn_label, _stop_turn_start, _stop_turn_end
    with _stop_turn_lock:
        _stop_turn_label = ""
        _stop_turn_start = 0.0
        _stop_turn_end = 0.0

# =======================
# Decision
# =======================
def _compute_decision_snapshot() -> Dict[str, Any]:
    global _last_back_ts, _back_until

    with lock:
        pts = list(latest_points)
        last_ts = latest_ts

    now = time.time()
    if not pts or (now - last_ts) > 2.0:
        _clear_stop_then_turn()
        return {"ok": False, "label": "STOP", "reason": "no_recent_lidar", "ts": now}

    corridor_info = _pick_best_heading_by_corridor(pts)
    clear0 = float(corridor_info["clear0_m"])
    best_h = float(corridor_info["best_heading_deg"])
    best_clear = float(corridor_info["best_clearance_m"])

    predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
    block_th = max(STOP_NEAR_M, predict_dist)

    # Debug distances
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
        "block_th_m": block_th,
        "robot_width_m": ROBOT_WIDTH_M,
        "clearance_margin_m": CLEARANCE_MARGIN_M,
        "stop_hold_sec": STOP_HOLD_SEC,
        "turn_hold_sec": TURN_HOLD_SEC,
        "back_hold_sec": BACK_HOLD_SEC,
        "back_cooldown_sec": BACK_COOLDOWN_SEC,
    }

    # 0) If we're in BACK hold window, keep backing
    with _behavior_lock:
        if now < _back_until:
            return {
                "ok": True,
                "label": "GO_BACK",
                "reason": "phase:BACK_HOLD",
                "ts": now,
                "dist": dist_pack,
                "threshold": thresh_pack,
                "corridor": corridor_info,
            }

    # 1) STOP->TURN chain: if in TURN window, return turn label
    turn_lbl = _check_stop_then_turn(now)
    if turn_lbl:
        # if suddenly front becomes super clear, cancel turning
        if clear0 > block_th * 1.25:
            _clear_stop_then_turn()
        else:
            return {
                "ok": True,
                "label": turn_lbl,
                "reason": "stop_then_turn:TURN",
                "ts": now,
                "dist": dist_pack,
                "threshold": thresh_pack,
                "corridor": corridor_info,
            }

    # 2) If front corridor is clearly safe => GO_STRAIGHT
    if clear0 > block_th * 1.10:
        _clear_stop_then_turn()
        return {
            "ok": True,
            "label": "GO_STRAIGHT",
            "reason": "front_corridor_clear",
            "ts": now,
            "dist": dist_pack,
            "threshold": thresh_pack,
            "corridor": corridor_info,
        }

    # Helpers for boxed/back logic
    boxed_side = (max(d_left_wide, d_diag_l) <= block_th) and (max(d_right_wide, d_diag_r) <= block_th)
    back_ok = d_back_wide > block_th

    # 3) Very near => prefer STOP then TURN (stable, tránh té)
    if clear0 <= STOP_NEAR_M:
        if abs(best_h) < 1e-6:
            next_turn = "TURN_LEFT" if d_left_wide >= d_right_wide else "TURN_RIGHT"
        else:
            next_turn = "TURN_LEFT" if best_h > 0 else "TURN_RIGHT"

        # only BACK if boxed and cooldown passed
        with _behavior_lock:
            if boxed_side and back_ok and (now - _last_back_ts) >= BACK_COOLDOWN_SEC:
                _last_back_ts = now
                _back_until = now + BACK_HOLD_SEC
                _clear_stop_then_turn()
                return {
                    "ok": True,
                    "label": "GO_BACK",
                    "reason": "very_near_boxed_back_once",
                    "ts": now,
                    "dist": dist_pack,
                    "threshold": thresh_pack,
                    "corridor": corridor_info,
                }

        # schedule TURN after STOP
        _arm_stop_then_turn(now, next_turn)
        return {
            "ok": True,
            "label": "STOP",
            "reason": "very_near_stop_then_turn",
            "ts": now,
            "dist": dist_pack,
            "threshold": thresh_pack,
            "corridor": corridor_info,
        }

    # 4) Not very near: if best corridor is passable, turn toward it (or keep forward if small)
    passable = (best_clear > block_th)
    if passable:
        if abs(best_h) <= 12.0:
            _clear_stop_then_turn()
            return {
                "ok": True,
                "label": "GO_STRAIGHT",
                "reason": "best_heading_small_keep_forward",
                "ts": now,
                "dist": dist_pack,
                "threshold": thresh_pack,
                "corridor": corridor_info,
            }
        lbl = "TURN_LEFT" if best_h > 0 else "TURN_RIGHT"
        _clear_stop_then_turn()
        return {
            "ok": True,
            "label": lbl,
            "reason": "front_blocked_turn_to_best_corridor",
            "ts": now,
            "dist": dist_pack,
            "threshold": thresh_pack,
            "corridor": corridor_info,
        }

    # 5) No passable corridor: STOP, but allow BACK with cooldown if boxed
    with _behavior_lock:
        if boxed_side and back_ok and (now - _last_back_ts) >= BACK_COOLDOWN_SEC:
            _last_back_ts = now
            _back_until = now + BACK_HOLD_SEC
            _clear_stop_then_turn()
            return {
                "ok": True,
                "label": "GO_BACK",
                "reason": "boxed_in_back_with_cooldown",
                "ts": now,
                "dist": dist_pack,
                "threshold": thresh_pack,
                "corridor": corridor_info,
            }

    _clear_stop_then_turn()
    return {
        "ok": True,
        "label": "STOP",
        "reason": "no_passable_corridor",
        "ts": now,
        "dist": dist_pack,
        "threshold": thresh_pack,
        "corridor": corridor_info,
    }

# =======================
# Points payload (compat autopilot)
# =======================
def _build_points_payload(limit: int = 1600) -> Dict[str, Any]:
    with lock:
        pts = list(latest_points[-limit:])
        ts = latest_ts

    out = []
    for (theta, dist_m, q, x, y, t) in pts:
        out.append({
            # original
            "theta": theta,
            "dist_m": dist_m,

            # compatibility:
            "angle": theta,
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

    with _stop_turn_lock:
        st = {"label": _stop_turn_label, "start": _stop_turn_start, "end": _stop_turn_end}

    with _behavior_lock:
        bh = {"last_back_ts": _last_back_ts, "back_until": _back_until}

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
        "robot_width_m": ROBOT_WIDTH_M,
        "corridor_half_w_m": (ROBOT_WIDTH_M / 2.0 + CLEARANCE_MARGIN_M),
        "lookahead_m": LOOKAHEAD_M,
        "stop_then_turn": st,
        "back_state": bh,
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
