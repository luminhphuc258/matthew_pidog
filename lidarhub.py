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

# Safety / motion model
STOP_NEAR_M = float(os.environ.get("STOP_NEAR_M", "0.30"))      # stop if center < 30cm
LOOKAHEAD_M = float(os.environ.get("LOOKAHEAD_M", "1.20"))      # obstacle counting range
PREDICT_T_SEC = float(os.environ.get("PREDICT_T_SEC", "1.0"))
ROBOT_SPEED_MPS = float(os.environ.get("ROBOT_SPEED_MPS", "0.35"))
SAFETY_MARGIN_M = float(os.environ.get("SAFETY_MARGIN_M", "0.10"))

# Plot frame / heading
FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))
FRONT_MIRROR = int(os.environ.get("FRONT_MIRROR", "0"))  # set 1 if left/right inverted

RECENT_SEC = float(os.environ.get("RECENT_SEC", "0.7"))

# ===== Auto front calib (back wall -> front = back + 180) =====
AUTO_FRONT_CALIB = int(os.environ.get("AUTO_FRONT_CALIB", "1"))
AUTO_CALIB_BIN_DEG = float(os.environ.get("AUTO_CALIB_BIN_DEG", "10.0"))
AUTO_CALIB_SMOOTH = float(os.environ.get("AUTO_CALIB_SMOOTH", "0.35"))
AUTO_CALIB_NEAR_CAP_M = float(os.environ.get("AUTO_CALIB_NEAR_CAP_M", "2.0"))
AUTO_CALIB_PERCENTILE = float(os.environ.get("AUTO_CALIB_PERCENTILE", "20.0"))

# ===== k=3 sectors in FRONT 180 =====
SECTOR_CENTER_DEG = float(os.environ.get("SECTOR_CENTER_DEG", "30.0"))  # center half-width (default +-30)
# RIGHT: [-90..-SECTOR_CENTER_DEG), CENTER: [-SECTOR_CENTER_DEG..+SECTOR_CENTER_DEG], LEFT: (+SECTOR_CENTER_DEG..+90]

FRONT_HALF_DEG = float(os.environ.get("FRONT_HALF_DEG", "90.0"))  # front 180 = [-90..+90]

# ===== Sticky turn behavior =====
# giữ TURN đến khi phía trước không còn vật cản < 40cm
CLEAR_RELEASE_M = float(os.environ.get("CLEAR_RELEASE_M", "0.40"))      # release when center_min > 40cm
CLEAR_CONFIRM_SEC = float(os.environ.get("CLEAR_CONFIRM_SEC", "0.25"))  # require stable clear a bit
TURN_STICKY_MIN_SEC = float(os.environ.get("TURN_STICKY_MIN_SEC", "0.35"))  # minimal hold to avoid flicker

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

_front_center_lock = threading.Lock()
_front_center_est_deg: float = FRONT_CENTER_DEG
_last_back_deg: Optional[float] = None

# Sticky turn state
_sticky_lock = threading.Lock()
_sticky_label: Optional[str] = None            # "TURN_LEFT" / "TURN_RIGHT" / None
_sticky_since: float = 0.0
_sticky_clear_start: Optional[float] = None


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

def _get_front_center_deg() -> float:
    with _front_center_lock:
        return float(_front_center_est_deg)

def _set_front_center_deg(v: float):
    global _front_center_est_deg
    with _front_center_lock:
        _front_center_est_deg = float(_wrap_deg(v))

def _rel_deg(theta_deg: float, center_deg: float) -> float:
    rel = _wrap_rel_deg(theta_deg - center_deg)  # [-180..180], + is left
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
# Auto FRONT calibration (back wall -> front)
# =======================
def _estimate_back_direction_percentile(
    pts: List[Tuple[float, float, int, float, float, float]],
    recent_sec: float
) -> Optional[float]:
    now = time.time()
    bin_deg = max(2.0, float(AUTO_CALIB_BIN_DEG))
    bins = max(12, int(round(360.0 / bin_deg)))

    recent = [(theta, dist_m, ts) for (theta, dist_m, q, x, y, ts) in pts if (now - ts) <= recent_sec and dist_m > 0.02]
    if len(recent) < 50:
        return None

    dists = sorted([d for (_, d, _) in recent])
    cutoff = _percentile(dists, float(AUTO_CALIB_PERCENTILE))
    cutoff = min(float(AUTO_CALIB_NEAR_CAP_M), cutoff)
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
# k=3 sector clustering in FRONT 180
# =======================
def _sector_name(rel_deg: float) -> Optional[str]:
    if abs(rel_deg) > FRONT_HALF_DEG:
        return None  # ignore back half
    c = float(SECTOR_CENTER_DEG)
    if rel_deg < -c:
        return "RIGHT"
    if rel_deg > c:
        return "LEFT"
    return "CENTER"

def _sector_stats(
    pts: List[Tuple[float, float, int, float, float, float]],
    front_abs: float
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      {
        "LEFT":   {"count": int, "min_dist": float},
        "CENTER": {"count": int, "min_dist": float},
        "RIGHT":  {"count": int, "min_dist": float},
      }
    count = number of points with dist < LOOKAHEAD_M
    min_dist = minimum distance seen in that sector (front half only)
    """
    now = time.time()
    out = {
        "LEFT":   {"count": 0.0, "min_dist": float("inf")},
        "CENTER": {"count": 0.0, "min_dist": float("inf")},
        "RIGHT":  {"count": 0.0, "min_dist": float("inf")},
    }

    for (theta, dist_m, q, x, y, ts) in pts:
        if now - ts > RECENT_SEC:
            continue
        if dist_m <= 0.02:
            continue

        rel = _rel_deg(theta, front_abs)
        sec = _sector_name(rel)
        if sec is None:
            continue

        if dist_m < out[sec]["min_dist"]:
            out[sec]["min_dist"] = dist_m

        if dist_m <= LOOKAHEAD_M:
            out[sec]["count"] += 1.0

    return out


# =======================
# Decision + Sticky Turn
# =======================
def _choose_direction_from_sectors(secs: Dict[str, Dict[str, float]]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Logic:
      - If CENTER min_dist <= STOP_NEAR_M => STOP
      - Else choose sector with smallest count. Tie-break:
          prefer CENTER, then the side with larger min_dist.
    """
    cmin = float(secs["CENTER"]["min_dist"])
    if cmin <= STOP_NEAR_M:
        return ("STOP", "center_too_close", {"center_min": cmin, "stop_near_m": STOP_NEAR_M})

    # counts
    lc = secs["LEFT"]["count"]
    cc = secs["CENTER"]["count"]
    rc = secs["RIGHT"]["count"]

    # find min count
    m = min(lc, cc, rc)
    candidates = []
    if lc == m: candidates.append("LEFT")
    if cc == m: candidates.append("CENTER")
    if rc == m: candidates.append("RIGHT")

    # prefer CENTER if tie
    if "CENTER" in candidates:
        best = "CENTER"
    else:
        # tie between LEFT/RIGHT => pick larger min_dist
        lmin = float(secs["LEFT"]["min_dist"])
        rmin = float(secs["RIGHT"]["min_dist"])
        best = "LEFT" if lmin >= rmin else "RIGHT"

    if best == "CENTER":
        return ("GO_STRAIGHT", "k3_sector_best_center", {"counts": {"L": lc, "C": cc, "R": rc}})
    if best == "LEFT":
        return ("TURN_LEFT", "k3_sector_best_left", {"counts": {"L": lc, "C": cc, "R": rc}})
    return ("TURN_RIGHT", "k3_sector_best_right", {"counts": {"L": lc, "C": cc, "R": rc}})

def _apply_sticky_turn(
    desired_label: str,
    center_min: float,
    now: float
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Sticky rule:
      - If currently sticky TURN_LEFT/RIGHT => keep it
        until center_min > CLEAR_RELEASE_M continuously for CLEAR_CONFIRM_SEC
      - Also enforce minimum hold time TURN_STICKY_MIN_SEC
    """
    global _sticky_label, _sticky_since, _sticky_clear_start

    with _sticky_lock:
        cur = _sticky_label

        # If currently sticky turn
        if cur in ("TURN_LEFT", "TURN_RIGHT"):
            held_for = now - _sticky_since

            # require min hold time
            if held_for < TURN_STICKY_MIN_SEC:
                return (cur, "sticky_min_hold", {"held_for": held_for})

            # check clear condition (center has no obstacle < 40cm)
            if center_min > CLEAR_RELEASE_M:
                if _sticky_clear_start is None:
                    _sticky_clear_start = now
                if (now - _sticky_clear_start) >= CLEAR_CONFIRM_SEC:
                    # release sticky
                    _sticky_label = None
                    _sticky_clear_start = None
                    cur = None
                else:
                    return (cur, "sticky_wait_clear_confirm", {"center_min": center_min, "clear_for": now - _sticky_clear_start})
            else:
                _sticky_clear_start = None
                return (cur, "sticky_turn_until_front_clear", {"center_min": center_min, "clear_release_m": CLEAR_RELEASE_M})

        # No sticky currently => if desired is a turn, set sticky
        if cur is None and desired_label in ("TURN_LEFT", "TURN_RIGHT"):
            _sticky_label = desired_label
            _sticky_since = now
            _sticky_clear_start = None
            return (desired_label, "set_sticky_turn", {"center_min": center_min})

    # default: allow desired
    return (desired_label, "not_sticky", {"center_min": center_min})

def _compute_decision() -> Dict[str, Any]:
    with lock:
        pts = list(latest_points)
        last_ts = float(latest_ts)

    now = time.time()
    if (not pts) or ((now - last_ts) > 2.0):
        return {"ok": False, "label": "STOP", "reason": "no_recent_lidar", "ts": now}

    # 1) auto calibrate front
    _auto_calibrate_front_center(pts)
    front_abs = _get_front_center_deg()

    # 2) build k=3 sector stats
    secs = _sector_stats(pts, front_abs)
    center_min = float(secs["CENTER"]["min_dist"])

    # 3) choose desired based on counts
    desired_label, desired_reason, desired_dbg = _choose_direction_from_sectors(secs)

    # 4) sticky turn apply
    final_label, sticky_reason, sticky_dbg = _apply_sticky_turn(desired_label, center_min, now)

    # 5) add predict threshold info (optional debug)
    predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M

    return {
        "ok": True,
        "label": final_label,
        "reason": f"{desired_reason} | {sticky_reason}",
        "ts": now,
        "debug": {
            "front_center_deg_used": float(front_abs),
            "back_deg_est": float(_last_back_deg) if _last_back_deg is not None else None,
            "sector_center_deg": float(SECTOR_CENTER_DEG),
            "lookahead_m": float(LOOKAHEAD_M),
            "stop_near_m": float(STOP_NEAR_M),
            "clear_release_m": float(CLEAR_RELEASE_M),
            "clear_confirm_sec": float(CLEAR_CONFIRM_SEC),
            "turn_sticky_min_sec": float(TURN_STICKY_MIN_SEC),
            "predict_dist_m": float(predict_dist),
            "sectors": {
                "LEFT":   {"count": int(secs["LEFT"]["count"]),   "min_dist": float(secs["LEFT"]["min_dist"])},
                "CENTER": {"count": int(secs["CENTER"]["count"]), "min_dist": float(secs["CENTER"]["min_dist"])},
                "RIGHT":  {"count": int(secs["RIGHT"]["count"]),  "min_dist": float(secs["RIGHT"]["min_dist"])},
            },
            "desired": {"label": desired_label, "reason": desired_reason, **desired_dbg},
            "sticky": {"label": final_label, "reason": sticky_reason, **sticky_dbg},
        }
    }


# =======================
# Points payload (adds angles for plotting)
# =======================
def _front_angles_for_plot(theta_raw: float, front_center_abs: float) -> Dict[str, Any]:
    rel = _rel_deg(theta_raw, front_center_abs)
    angle_front_360 = _wrap_deg(rel)  # 0..360 where 0=front
    out = {
        "rel_deg": float(rel),
        "angle_front_360": float(angle_front_360),
        "front_0_180": None,
        "is_front_180": bool(abs(rel) <= FRONT_HALF_DEG),
        "k3_sector": None,
    }
    sec = _sector_name(rel)
    out["k3_sector"] = sec
    if abs(rel) <= FRONT_HALF_DEG:
        out["front_0_180"] = float(rel + 90.0)  # -90..+90 -> 0..180 (90=front)
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
            "angle": theta,
            "dist_m": dist_m,
            "dist_cm": dist_m * 100.0,
            "q": q,
            "x": x,
            "y": y,
            "ts": t,
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

    with _sticky_lock:
        sticky = _sticky_label
        sticky_since = _sticky_since
        clear_start = _sticky_clear_start

    now = time.time()
    return jsonify({
        "running": status["running"],
        "port": status["port"],
        "baud": status["baud"],
        "bin": status["bin"],
        "points_buffered": pts_n,
        "last_point_ts": ts,
        "age_s": (now - ts) if ts else None,
        "last_error": status["last_error"],
        "pid": proc.pid if proc else None,

        "latest_label": lbl,
        "front_center_deg_used": float(_get_front_center_deg()),
        "back_deg_est": float(_last_back_deg) if _last_back_deg is not None else None,
        "auto_front_calib": int(AUTO_FRONT_CALIB),
        "auto_percentile": float(AUTO_CALIB_PERCENTILE),
        "auto_cap_m": float(AUTO_CALIB_NEAR_CAP_M),
        "front_mirror": int(FRONT_MIRROR),

        "k3": {
            "sector_center_deg": float(SECTOR_CENTER_DEG),
            "lookahead_m": float(LOOKAHEAD_M),
            "stop_near_m": float(STOP_NEAR_M),
        },

        "sticky_turn": {
            "sticky_label": sticky,
            "held_for_sec": (now - sticky_since) if sticky else None,
            "clear_for_sec": (now - clear_start) if clear_start else None,
            "clear_release_m": float(CLEAR_RELEASE_M),
            "clear_confirm_sec": float(CLEAR_CONFIRM_SEC),
            "turn_sticky_min_sec": float(TURN_STICKY_MIN_SEC),
        },

        "decision": dec,
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
