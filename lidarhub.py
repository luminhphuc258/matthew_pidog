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

THROTTLE_MS = int(os.environ.get("THROTTLE_MS", "250"))
THROTTLE_S = THROTTLE_MS / 1000.0

STOP_NEAR_M = float(os.environ.get("STOP_NEAR_M", "0.30"))
PREDICT_T_SEC = float(os.environ.get("PREDICT_T_SEC", "1.0"))
ROBOT_SPEED_MPS = float(os.environ.get("ROBOT_SPEED_MPS", "0.35"))
SAFETY_MARGIN_M = float(os.environ.get("SAFETY_MARGIN_M", "0.10"))

FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))
FRONT_MIRROR = int(os.environ.get("FRONT_MIRROR", "0"))

FRONT_WIDTH_DEG = float(os.environ.get("FRONT_WIDTH_DEG", "30.0"))
WIDE_WIDTH_DEG  = float(os.environ.get("WIDE_WIDTH_DEG", "70.0"))

ROBOT_WIDTH_M = float(os.environ.get("ROBOT_WIDTH_M", "0.15"))
CLEARANCE_MARGIN_M = float(os.environ.get("CLEARANCE_MARGIN_M", "0.03"))
LOOKAHEAD_M = float(os.environ.get("LOOKAHEAD_M", "1.20"))

FRONT_HALF_DEG = float(os.environ.get("FRONT_HALF_DEG", "90.0"))  # front 180 = [-90..+90]

CLUSTER_ANG_GAP_DEG = float(os.environ.get("CLUSTER_ANG_GAP_DEG", "2.5"))
CLUSTER_DIST_JUMP_M = float(os.environ.get("CLUSTER_DIST_JUMP_M", "0.25"))
CLUSTER_MIN_PTS     = int(os.environ.get("CLUSTER_MIN_PTS", "4"))
RECENT_SEC          = float(os.environ.get("RECENT_SEC", "0.7"))

# ===== Auto front calib (FIXED) =====
AUTO_FRONT_CALIB    = int(os.environ.get("AUTO_FRONT_CALIB", "1"))
AUTO_CALIB_BIN_DEG  = float(os.environ.get("AUTO_CALIB_BIN_DEG", "10.0"))
AUTO_CALIB_SMOOTH   = float(os.environ.get("AUTO_CALIB_SMOOTH", "0.35"))
# IMPORTANT: dùng percentile nên ngưỡng này chỉ là cap, đặt lớn 1 chút cho chắc
AUTO_CALIB_NEAR_CAP_M = float(os.environ.get("AUTO_CALIB_NEAR_CAP_M", "2.0"))
AUTO_CALIB_PERCENTILE = float(os.environ.get("AUTO_CALIB_PERCENTILE", "20.0"))  # 20% điểm gần nhất

STOP_HOLD_SEC = float(os.environ.get("STOP_HOLD_SEC", "0.35"))
TURN_HOLD_SEC = float(os.environ.get("TURN_HOLD_SEC", "2.0"))
TURN_HOLD_SEC = max(2.0, TURN_HOLD_SEC)

LINE_RE = re.compile(r"theta:\s*([0-9.]+)\s+Dist:\s*([0-9.]+)\s+Q:\s*(\d+)")

app = Flask(__name__)

lock = threading.Lock()
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

cache_lock = threading.Lock()
_points_cache: Dict[str, Any] = {"ts": 0.0, "payload": None}
_decision_cache: Dict[str, Any] = {"ts": 0.0, "payload": None}

decision_state_lock = threading.Lock()
latest_decision_label: str = "STOP"
latest_decision_full: Dict[str, Any] = {"ok": False, "label": "STOP", "reason": "init", "ts": 0.0}

_turn_lock = threading.Lock()
_phase: str = "NORMAL"
_phase_until: float = 0.0
_turn_label: str = ""

_front_center_lock = threading.Lock()
_front_center_est_deg: float = FRONT_CENTER_DEG
_last_back_deg: Optional[float] = None


# =======================
# Math helpers
# =======================
def _wrap_deg(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def _wrap_rel_deg(a: float) -> float:
    a = _wrap_deg(a)
    if a > 180.0:
        a -= 360.0
    return a

def _ang_dist_deg(a: float, b: float) -> float:
    d = abs(_wrap_deg(a) - _wrap_deg(b))
    return min(d, 360.0 - d)

def _ema_angle_deg(prev_deg: float, new_deg: float, alpha: float) -> float:
    prev = _wrap_deg(prev_deg)
    new = _wrap_deg(new_deg)
    delta = _wrap_rel_deg(new - prev)
    return _wrap_deg(prev + alpha * delta)

def polar_to_xy_m(theta_deg: float, dist_m: float) -> Tuple[float, float]:
    th = math.radians(theta_deg)
    return dist_m * math.cos(th), dist_m * math.sin(th)

def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("inf")
    p = max(0.0, min(100.0, p))
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

def _required_gap_width_m() -> float:
    return float(ROBOT_WIDTH_M + 2.0 * CLEARANCE_MARGIN_M)

def _get_front_center_deg() -> float:
    with _front_center_lock:
        return float(_front_center_est_deg)

def _set_front_center_deg(v: float):
    global _front_center_est_deg
    with _front_center_lock:
        _front_center_est_deg = float(_wrap_deg(v))

def _rel_deg(theta_deg: float, center_deg: float) -> float:
    rel = _wrap_rel_deg(theta_deg - center_deg)
    if FRONT_MIRROR == 1:
        rel = -rel
    return rel


# =======================
# Lidar process
# =======================
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
        status["last_error"] = "ultra_simple has no stdout"
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
            x, y = polar_to_xy_m(theta, dist_m)

            with lock:
                latest_points.append((theta, dist_m, q, x, y, ts))
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


# =======================
# Auto FRONT calibration (FIXED for your data)
# =======================
def _estimate_back_direction_percentile(
    pts: List[Tuple[float, float, int, float, float, float]],
    recent_sec: float
) -> Optional[float]:
    """
    - Lấy các điểm recent
    - Lấy distance cutoff = percentile(P) (vd 20%) và cap bởi AUTO_CALIB_NEAR_CAP_M
    - Histogram theo theta bin
    - Bin có nhiều điểm gần nhất + mean_d nhỏ => back (tường)
    """
    now = time.time()
    bin_deg = max(2.0, float(AUTO_CALIB_BIN_DEG))
    bins = max(12, int(round(360.0 / bin_deg)))

    recent = [(theta, dist_m, ts) for (theta, dist_m, q, x, y, ts) in pts if (now - ts) <= recent_sec and dist_m > 0.02]
    if len(recent) < 50:
        return None

    dists = sorted([d for (_, d, _) in recent])
    cutoff = _percentile(dists, float(AUTO_CALIB_PERCENTILE))
    cutoff = min(float(AUTO_CALIB_NEAR_CAP_M), cutoff)
    # Nếu cutoff quá nhỏ (trường hợp dữ liệu sạch), nâng lên chút cho chắc
    cutoff = max(0.35, cutoff)

    counts = [0] * bins
    dist_sums = [0.0] * bins

    for (theta, dist_m, ts) in recent:
        if dist_m > cutoff:
            continue
        idx = int(_wrap_deg(theta) // bin_deg) % bins
        counts[idx] += 1
        dist_sums[idx] += dist_m

    best_idx = None
    best_score = -1e18
    for i in range(bins):
        if counts[i] < 6:
            continue
        mean_d = dist_sums[i] / max(1, counts[i])
        # score: ưu tiên gần + nhiều điểm
        score = (counts[i] * 1.0) + (2.0 - mean_d) * 6.0
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    back_center = (best_idx + 0.5) * bin_deg
    return float(_wrap_deg(back_center))

def _auto_calibrate_front_center(pts: List[Tuple[float, float, int, float, float, float]]):
    global _last_back_deg
    if AUTO_FRONT_CALIB != 1:
        _set_front_center_deg(FRONT_CENTER_DEG)
        _last_back_deg = None
        return

    back = _estimate_back_direction_percentile(pts, recent_sec=RECENT_SEC)
    if back is None:
        return

    _last_back_deg = back
    front = _wrap_deg(back + 180.0)

    prev = _get_front_center_deg()
    alpha = max(0.05, min(1.0, float(AUTO_CALIB_SMOOTH)))
    sm = _ema_angle_deg(prev, front, alpha)
    _set_front_center_deg(sm)


# =======================
# Sector + Decision (simple + reliable)
# =======================
def _min_dist_sector_abs(
    pts: List[Tuple[float, float, int, float, float, float]],
    center_abs_deg: float,
    width_deg: float
) -> float:
    now = time.time()
    half = width_deg / 2.0
    best = float("inf")
    for (theta, dist_m, q, x, y, ts) in pts:
        if now - ts > RECENT_SEC:
            continue
        if _ang_dist_deg(theta, center_abs_deg) <= half:
            if dist_m < best:
                best = dist_m
    return best

def _compute_decision() -> Dict[str, Any]:
    with lock:
        pts = list(latest_points)
        last_ts = float(latest_ts)

    now = time.time()
    if (not pts) or ((now - last_ts) > 2.0):
        return {"ok": False, "label": "STOP", "reason": "no_recent_lidar", "ts": now}

    # 1) FIX FRONT from your scenario (back wall)
    _auto_calibrate_front_center(pts)
    front_abs = _get_front_center_deg()
    back_abs = _wrap_deg(front_abs + 180.0)  # just for debug

    # 2) check front distances
    front_narrow = _min_dist_sector_abs(pts, front_abs, FRONT_WIDTH_DEG)
    front_wide   = _min_dist_sector_abs(pts, front_abs, WIDE_WIDTH_DEG)

    predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
    block_th = max(STOP_NEAR_M, predict_dist)

    # RULE A: nếu phía trước rộng & hẹp đều xa hơn LOOKAHEAD => GO_STRAIGHT
    # (trường hợp của bạn: trước trống)
    if (front_narrow > LOOKAHEAD_M) and (front_wide > LOOKAHEAD_M):
        return {
            "ok": True,
            "label": "GO_STRAIGHT",
            "reason": "front_clear_by_sector(auto_front_calib)",
            "ts": now,
            "debug": {
                "front_center_deg_used": float(front_abs),
                "back_deg_est": float(_last_back_deg) if _last_back_deg is not None else None,
                "front_narrow_m": float(front_narrow),
                "front_wide_m": float(front_wide),
                "lookahead_m": float(LOOKAHEAD_M),
            }
        }

    # RULE B: quá gần => STOP
    if front_narrow <= STOP_NEAR_M:
        return {
            "ok": True,
            "label": "STOP",
            "reason": "very_near_obstacle_front",
            "ts": now,
            "debug": {
                "front_center_deg_used": float(front_abs),
                "front_narrow_m": float(front_narrow),
                "stop_near_m": float(STOP_NEAR_M),
            }
        }

    # RULE C: có vật cản trong tầm block => TURN theo bên nào thoáng hơn
    if front_narrow <= block_th or front_wide <= block_th:
        left_abs  = _wrap_deg(front_abs + 90.0)
        right_abs = _wrap_deg(front_abs - 90.0)
        left_d  = _min_dist_sector_abs(pts, left_abs,  WIDE_WIDTH_DEG)
        right_d = _min_dist_sector_abs(pts, right_abs, WIDE_WIDTH_DEG)
        lbl = "TURN_LEFT" if left_d >= right_d else "TURN_RIGHT"
        return {
            "ok": True,
            "label": lbl,
            "reason": "blocked_front_turn_to_clearer_side",
            "ts": now,
            "debug": {
                "front_center_deg_used": float(front_abs),
                "front_narrow_m": float(front_narrow),
                "front_wide_m": float(front_wide),
                "block_th_m": float(block_th),
                "left_m": float(left_d),
                "right_m": float(right_d),
            }
        }

    # default
    return {
        "ok": True,
        "label": "GO_STRAIGHT",
        "reason": "default_forward",
        "ts": now,
        "debug": {
            "front_center_deg_used": float(front_abs),
            "front_narrow_m": float(front_narrow),
            "front_wide_m": float(front_wide),
        }
    }


# =======================
# Points payload (adds angles for your 0-180 plot)
# =======================
def _front_angles_for_plot(theta_raw: float, front_center_abs: float) -> Dict[str, Any]:
    """
    angle_front_360:
      0   = FRONT
      90  = LEFT
      180 = BACK
      270 = RIGHT

    front_0_180 (only if in front 180deg):
      0   = RIGHT
      90  = FRONT
      180 = LEFT
    """
    rel = _rel_deg(theta_raw, front_center_abs)  # [-180..180], + is left
    angle_front_360 = _wrap_deg(rel)             # 0..360 where 0=front
    out = {
        "rel_deg": float(rel),
        "angle_front_360": float(angle_front_360),
        "front_0_180": None,
        "is_front_180": bool(abs(rel) <= FRONT_HALF_DEG),
    }
    if abs(rel) <= FRONT_HALF_DEG:
        # map rel [-90..+90] -> front_0_180 [0..180] with 90=front
        # rel=-90 (right) -> 0, rel=0 -> 90, rel=+90 (left) -> 180
        out["front_0_180"] = float(rel + 90.0)
    return out

def _build_points_payload(limit: int = 1600) -> Dict[str, Any]:
    with lock:
        pts = list(latest_points[-limit:])
        ts = float(latest_ts)

    front_abs = _get_front_center_deg()

    out = []
    for (theta, dist_m, q, x, y, t) in pts:
        ang = _front_angles_for_plot(theta, front_abs)
        out.append({
            "theta": theta,
            "angle": theta,               # compat
            "dist_m": dist_m,
            "dist_cm": dist_m * 100.0,    # compat
            "q": q,
            "x": x,
            "y": y,
            "ts": t,
            # new for plotting / debug
            **ang
        })

    return {
        "ok": True,
        "ts": time.time(),
        "last_point_ts": ts,
        "n": len(out),
        "points": out,
        "frame": {
            "front_center_deg_used": float(front_abs),
            "back_deg_est": float(_last_back_deg) if _last_back_deg is not None else None,
            "max_range_m": float(MAX_RANGE_M),
        }
    }


# =======================
# Workers
# =======================
def decision_worker():
    global latest_decision_label, latest_decision_full
    while True:
        payload = _compute_decision()
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
        ts = float(latest_ts)

    with decision_state_lock:
        lbl = latest_decision_label
        dec = dict(latest_decision_full)

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
        "latest_label": lbl,
        "front_center_deg_used": float(_get_front_center_deg()),
        "back_deg_est": float(_last_back_deg) if _last_back_deg is not None else None,
        "auto_front_calib": int(AUTO_FRONT_CALIB),
        "auto_percentile": float(AUTO_CALIB_PERCENTILE),
        "auto_cap_m": float(AUTO_CALIB_NEAR_CAP_M),
        "front_mirror": int(FRONT_MIRROR),
        "decision": dec,
        "note": "Auto front calib uses nearest-distance percentile (fixes your case: back wall ~0.9m). 0deg is FRONT after calib.",
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
        payload = _compute_decision()
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
    _set_front_center_deg(FRONT_CENTER_DEG)
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    threading.Thread(target=points_worker, daemon=True).start()
    threading.Thread(target=decision_worker, daemon=True).start()
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
