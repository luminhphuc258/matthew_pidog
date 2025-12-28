#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import math
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
POINT_LIMIT = int(os.environ.get("POINT_LIMIT", "3000"))

THROTTLE_MS = int(os.environ.get("THROTTLE_MS", "200"))
THROTTLE_S = THROTTLE_MS / 1000.0

# Safety / motion model
STOP_NEAR_M = float(os.environ.get("STOP_NEAR_M", "0.30"))      # used by k3 center stop condition
LOOKAHEAD_M = float(os.environ.get("LOOKAHEAD_M", "1.20"))
PREDICT_T_SEC = float(os.environ.get("PREDICT_T_SEC", "1.0"))
ROBOT_SPEED_MPS = float(os.environ.get("ROBOT_SPEED_MPS", "0.35"))
SAFETY_MARGIN_M = float(os.environ.get("SAFETY_MARGIN_M", "0.10"))

# Plot/Heading
FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))
FRONT_MIRROR = int(os.environ.get("FRONT_MIRROR", "0"))  # 1 if left/right inverted
RECENT_SEC = float(os.environ.get("RECENT_SEC", "0.7"))

# Auto front calib
AUTO_FRONT_CALIB = int(os.environ.get("AUTO_FRONT_CALIB", "1"))
AUTO_CALIB_BIN_DEG = float(os.environ.get("AUTO_CALIB_BIN_DEG", "10.0"))
AUTO_CALIB_SMOOTH = float(os.environ.get("AUTO_CALIB_SMOOTH", "0.35"))
AUTO_CALIB_NEAR_CAP_M = float(os.environ.get("AUTO_CALIB_NEAR_CAP_M", "2.0"))
AUTO_CALIB_PERCENTILE = float(os.environ.get("AUTO_CALIB_PERCENTILE", "20.0"))

# k=3 sectors (front 180)
SECTOR_CENTER_DEG = float(os.environ.get("SECTOR_CENTER_DEG", "30.0"))
FRONT_HALF_DEG = float(os.environ.get("FRONT_HALF_DEG", "90.0"))

# ===== Front arc safety override =====
FRONT_ARC_DEG = float(os.environ.get("FRONT_ARC_DEG", "85.0"))              # ±85°
FRONT_ARC_BLOCK_M = float(os.environ.get("FRONT_ARC_BLOCK_M", "0.55"))      # block GO_STRAIGHT if <= 0.55m
FRONT_ARC_HARD_STOP_M = float(os.environ.get("FRONT_ARC_HARD_STOP_M", "0.35"))  # STOP one-shot if <= 0.35m

# Re-arm threshold: must clear past this to allow next STOP again
HARD_STOP_REARM_M = float(os.environ.get("HARD_STOP_REARM_M", "0.45"))

# ===== Heading Lock (anti-oscillation) =====
HEADING_LOCK_ENABLE = int(os.environ.get("HEADING_LOCK_ENABLE", "1"))
HEADING_DEADBAND_DEG = float(os.environ.get("HEADING_DEADBAND_DEG", "8.0"))   # small error -> go straight
AVOID_TURN_STEP_DEG = float(os.environ.get("AVOID_TURN_STEP_DEG", "35.0"))     # set new target yaw when blocked
MIN_AVOID_HOLD_S = float(os.environ.get("MIN_AVOID_HOLD_S", "0.7"))            # avoid mode minimal time
CLEAR_TO_TRACK_M = float(os.environ.get("CLEAR_TO_TRACK_M", "0.65"))           # must be > block a bit to return to TRACK
CENTER_CLEAR_M = float(os.environ.get("CENTER_CLEAR_M", "0.60"))               # center min distance to consider forward safe

# ===== Map (occupancy) =====
MAP_SIZE_M = float(os.environ.get("MAP_SIZE_M", "10.0"))
MAP_RES_M = float(os.environ.get("MAP_RES_M", "0.05"))
MAP_DECAY = float(os.environ.get("MAP_DECAY", "0.985"))
MAP_HIT = float(os.environ.get("MAP_HIT", "30.0"))
MAP_MAX = float(os.environ.get("MAP_MAX", "255.0"))

# pose dead-reckoning (based on decision)
TURN_RATE_DEG_S = float(os.environ.get("TURN_RATE_DEG_S", "70.0"))
BACK_SPEED_MPS = float(os.environ.get("BACK_SPEED_MPS", "0.20"))

# view render
VIEW_SIZE_PX = int(os.environ.get("VIEW_SIZE_PX", "640"))
VIEW_RANGE_M = float(os.environ.get("VIEW_RANGE_M", "3.5"))
SECTOR_RING_M = float(os.environ.get("SECTOR_RING_M", "1.2"))
SECTOR_ALPHA = int(os.environ.get("SECTOR_ALPHA", "90"))

LINE_RE = re.compile(r"theta:\s*([0-9.]+)\s+Dist:\s*([0-9.]+)\s+Q:\s*(\d+)")

app = Flask(__name__)

lock = threading.Lock()
latest_points: List[Tuple[float, float, int, float, float, float]] = []  # (theta, dist_m, q, x, y, ts)
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
_map_cache: Dict[str, Any] = {"ts": 0.0, "png": None}

decision_state_lock = threading.Lock()
latest_decision_label: str = "STOP"
latest_decision_full: Dict[str, Any] = {"ok": False, "label": "STOP", "reason": "init", "ts": 0.0}

_front_center_lock = threading.Lock()
_front_center_est_deg: float = FRONT_CENTER_DEG
_last_back_deg: Optional[float] = None

# ===== STOP one-shot then TURN state =====
_stopturn_lock = threading.Lock()
_stopturn_active: bool = False
_stopturn_next: Optional[str] = None         # "TURN_LEFT"/"TURN_RIGHT"
_stopturn_stop_issued: bool = False
_stopturn_reason: str = ""
_hard_stop_armed: bool = True               # re-arm gate to avoid repeated STOP spam

# ===== Heading lock state =====
_nav_lock = threading.Lock()
_nav_mode: str = "TRACK"   # TRACK | AVOID
_nav_target_yaw: Optional[float] = None
_nav_set_ts: float = 0.0

# Pose for map (world frame)
_pose_lock = threading.Lock()
pose_x = 0.0
pose_y = 0.0
pose_yaw = 0.0  # radians, 0 = facing +X in world
_pose_last_ts = 0.0

# Occupancy grid
_grid_lock = threading.Lock()
grid_w = int(round(MAP_SIZE_M / MAP_RES_M))
grid_h = int(round(MAP_SIZE_M / MAP_RES_M))
grid = [[0.0 for _ in range(grid_w)] for __ in range(grid_h)]


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

def _ema_angle_deg(prev_deg: float, new_deg: float, alpha: float) -> float:
    prev = _wrap_deg(prev_deg)
    new = _wrap_deg(new_deg)
    delta = _wrap_rel_deg(new - prev)
    return _wrap_deg(prev + alpha * delta)

def _wrap_pi(rad: float) -> float:
    while rad > math.pi:
        rad -= 2 * math.pi
    while rad < -math.pi:
        rad += 2 * math.pi
    return rad

def _angle_err(target: float, current: float) -> float:
    return _wrap_pi(target - current)

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
    # rel in [-180..180], + is left
    rel = _wrap_rel_deg(theta_deg - center_deg)
    if FRONT_MIRROR == 1:
        rel = -rel
    return rel

def _sector_name(rel_deg: float) -> Optional[str]:
    if abs(rel_deg) > FRONT_HALF_DEG:
        return None
    c = float(SECTOR_CENTER_DEG)
    if rel_deg < -c:
        return "RIGHT"
    if rel_deg > c:
        return "LEFT"
    return "CENTER"


# =======================
# STOP->TURN helpers (one-shot stop)
# =======================
def _start_stop_then_turn(turn_label: str, reason: str):
    global _stopturn_active, _stopturn_next, _stopturn_stop_issued, _stopturn_reason, _hard_stop_armed
    with _stopturn_lock:
        _stopturn_active = True
        _stopturn_next = turn_label
        _stopturn_stop_issued = False
        _stopturn_reason = reason
        _hard_stop_armed = False  # disarm until cleared

def _stopturn_clear():
    global _stopturn_active, _stopturn_next, _stopturn_stop_issued, _stopturn_reason
    with _stopturn_lock:
        _stopturn_active = False
        _stopturn_next = None
        _stopturn_stop_issued = False
        _stopturn_reason = ""

def _stopturn_get_state() -> Tuple[bool, Optional[str], bool, str]:
    with _stopturn_lock:
        return _stopturn_active, _stopturn_next, _stopturn_stop_issued, _stopturn_reason

def _stopturn_mark_stop_issued():
    global _stopturn_stop_issued
    with _stopturn_lock:
        _stopturn_stop_issued = True

def _is_hard_stop_armed() -> bool:
    with _stopturn_lock:
        return bool(_hard_stop_armed)

def _rearm_hard_stop():
    global _hard_stop_armed
    with _stopturn_lock:
        _hard_stop_armed = True


# =======================
# Heading lock helpers
# =======================
def _nav_get() -> Tuple[str, Optional[float], float]:
    with _nav_lock:
        return _nav_mode, _nav_target_yaw, _nav_set_ts

def _nav_set(mode: str, target_yaw: Optional[float], reason: str = ""):
    global _nav_mode, _nav_target_yaw, _nav_set_ts
    with _nav_lock:
        _nav_mode = mode
        _nav_target_yaw = target_yaw
        _nav_set_ts = time.time()

def _nav_ensure_target_from_pose():
    global _nav_target_yaw
    with _nav_lock:
        if _nav_target_yaw is None:
            with _pose_lock:
                _nav_target_yaw = float(pose_yaw)
            _nav_set_ts = time.time()


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
# Auto FRONT calibration
# =======================
def _estimate_back_direction_percentile(
    pts: List[Tuple[float, float, int, float, float, float]],
    recent_sec: float
) -> Optional[float]:
    now = time.time()
    bin_deg = max(2.0, float(AUTO_CALIB_BIN_DEG))
    bins = max(12, int(round(360.0 / bin_deg)))

    recent = [(theta, dist_m, ts) for (theta, dist_m, q, x, y, ts) in pts
              if (now - ts) <= recent_sec and dist_m > 0.02]
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
# k=3 sector stats
# =======================
def _sector_stats(
    pts: List[Tuple[float, float, int, float, float, float]],
    front_abs: float
) -> Dict[str, Dict[str, float]]:
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

def _choose_turn_to_open_side(secs: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, Any]]:
    lmin = float(secs["LEFT"]["min_dist"])
    rmin = float(secs["RIGHT"]["min_dist"])
    lc = float(secs["LEFT"]["count"])
    rc = float(secs["RIGHT"]["count"])

    L = 999.0 if not math.isfinite(lmin) else lmin
    R = 999.0 if not math.isfinite(rmin) else rmin

    if abs(L - R) > 1e-6:
        lbl = "TURN_LEFT" if L > R else "TURN_RIGHT"
    else:
        lbl = "TURN_LEFT" if lc <= rc else "TURN_RIGHT"

    return lbl, {"left": {"min": lmin, "count": lc}, "right": {"min": rmin, "count": rc}}

def _front_arc_min(
    pts: List[Tuple[float, float, int, float, float, float]],
    front_abs: float
) -> Tuple[float, Optional[float]]:
    now = time.time()
    best_d = float("inf")
    best_rel = None
    arc = float(FRONT_ARC_DEG)

    for (theta, dist_m, q, x, y, ts) in pts:
        if now - ts > RECENT_SEC:
            continue
        if dist_m <= 0.02:
            continue
        rel = _rel_deg(theta, front_abs)
        if abs(rel) > arc:
            continue
        if dist_m < best_d:
            best_d = dist_m
            best_rel = rel

    return best_d, best_rel


# =======================
# Decision (with Heading Lock)
# =======================
def _compute_decision() -> Dict[str, Any]:
    with lock:
        pts = list(latest_points)
        last_ts = float(latest_ts)

    now = time.time()
    if (not pts) or ((now - last_ts) > 2.0):
        return {"ok": False, "label": "STOP", "reason": "no_recent_lidar", "ts": now}

    _auto_calibrate_front_center(pts)
    front_abs = _get_front_center_deg()

    secs = _sector_stats(pts, front_abs)
    center_min = float(secs["CENTER"]["min_dist"])

    fa_min, fa_rel = _front_arc_min(pts, front_abs)
    hard_stop = (fa_min <= float(FRONT_ARC_HARD_STOP_M))
    block_go = (fa_min <= float(FRONT_ARC_BLOCK_M))

    # re-arm when clearly safe again
    if (not hard_stop) and (fa_min > float(HARD_STOP_REARM_M)):
        if not _is_hard_stop_armed():
            _rearm_hard_stop()

    # Basic "front clear" condition (for TRACK mode)
    front_clear = (fa_min > max(float(FRONT_ARC_BLOCK_M), float(CLEAR_TO_TRACK_M))) and (center_min > float(CENTER_CLEAR_M))

    # =========================
    # 1) STOP->TURN sequence (one-shot stop)
    # =========================
    st_active, st_next, st_stop_issued, st_reason = _stopturn_get_state()
    if st_active and st_next in ("TURN_LEFT", "TURN_RIGHT"):
        # exit if clear
        if front_clear:
            _stopturn_clear()
        else:
            if not st_stop_issued:
                _stopturn_mark_stop_issued()
                label = "STOP"
                reason = f"stop_once_then_turn | {st_reason}"
            else:
                label = st_next
                reason = f"turn_after_stop_once | {st_reason}"

            # When stopturn is turning -> also set nav to AVOID and set a target yaw stable
            if HEADING_LOCK_ENABLE == 1:
                with _pose_lock:
                    cyaw = float(pose_yaw)
                step = math.radians(AVOID_TURN_STEP_DEG)
                target = cyaw + (step if label == "TURN_LEFT" else -step)
                _nav_set("AVOID", _wrap_pi(target), reason="stopturn->avoid")

            predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
            return {
                "ok": True,
                "label": label,
                "reason": reason,
                "ts": now,
                "debug": {
                    "front_center_deg_used": float(front_abs),
                    "back_deg_est": float(_last_back_deg) if _last_back_deg is not None else None,
                    "sectors": {
                        "LEFT":   {"count": int(secs["LEFT"]["count"]),   "min_dist": float(secs["LEFT"]["min_dist"])},
                        "CENTER": {"count": int(secs["CENTER"]["count"]), "min_dist": float(secs["CENTER"]["min_dist"])},
                        "RIGHT":  {"count": int(secs["RIGHT"]["count"]),  "min_dist": float(secs["RIGHT"]["min_dist"])},
                    },
                    "predict_dist_m": float(predict_dist),
                    "front_arc": {
                        "deg": float(FRONT_ARC_DEG),
                        "block_m": float(FRONT_ARC_BLOCK_M),
                        "hard_stop_m": float(FRONT_ARC_HARD_STOP_M),
                        "rearm_m": float(HARD_STOP_REARM_M),
                        "min_dist": float(fa_min),
                        "min_rel_deg": float(fa_rel) if fa_rel is not None else None,
                    },
                    "stop_then_turn": {
                        "active": True,
                        "stop_issued": bool(st_stop_issued),
                        "next": st_next,
                        "armed": _is_hard_stop_armed(),
                    }
                }
            }

    # =========================
    # 2) Trigger STOP->TURN if hard_stop AND armed
    # =========================
    if hard_stop and _is_hard_stop_armed():
        turn_lbl, turn_dbg = _choose_turn_to_open_side(secs)
        _start_stop_then_turn(turn_lbl, f"front_arc_hard_stop(min={fa_min:.3f}, rel={fa_rel}) -> {turn_lbl}")

        predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
        return {
            "ok": True,
            "label": "STOP",
            "reason": "front_arc_hard_stop_stop_once_then_turn",
            "ts": now,
            "debug": {
                "front_center_deg_used": float(front_abs),
                "back_deg_est": float(_last_back_deg) if _last_back_deg is not None else None,
                "sectors": {
                    "LEFT":   {"count": int(secs["LEFT"]["count"]),   "min_dist": float(secs["LEFT"]["min_dist"])},
                    "CENTER": {"count": int(secs["CENTER"]["count"]), "min_dist": float(secs["CENTER"]["min_dist"])},
                    "RIGHT":  {"count": int(secs["RIGHT"]["count"]),  "min_dist": float(secs["RIGHT"]["min_dist"])},
                },
                "predict_dist_m": float(predict_dist),
                "front_arc": {
                    "deg": float(FRONT_ARC_DEG),
                    "block_m": float(FRONT_ARC_BLOCK_M),
                    "hard_stop_m": float(FRONT_ARC_HARD_STOP_M),
                    "rearm_m": float(HARD_STOP_REARM_M),
                    "min_dist": float(fa_min),
                    "min_rel_deg": float(fa_rel) if fa_rel is not None else None,
                },
                "stop_then_turn": {
                    "active": True,
                    "stop_issued": False,
                    "next": turn_lbl,
                    "turn_debug": turn_dbg,
                    "armed": _is_hard_stop_armed(),
                }
            }
        }

    # =========================
    # 3) Heading Lock navigation
    # =========================
    if HEADING_LOCK_ENABLE == 1:
        _nav_ensure_target_from_pose()
        mode, target, set_ts = _nav_get()
        with _pose_lock:
            cyaw = float(pose_yaw)

        if target is None:
            target = cyaw
            _nav_set("TRACK", target, reason="init_target")

        # If clear -> TRACK and lock to current yaw (keeps direction stable)
        if front_clear:
            if mode != "TRACK":
                _nav_set("TRACK", cyaw, reason="clear->track")
                mode, target, set_ts = _nav_get()
            else:
                # slowly refresh target toward current yaw to reduce drift
                # (very small update)
                alpha = 0.10
                new_target = _wrap_pi(target + alpha * _angle_err(cyaw, target))
                _nav_set("TRACK", new_target, reason="track_refresh")
                mode, target, set_ts = _nav_get()
        else:
            # blocked -> AVOID: choose open side once, set target yaw and keep it (anti-oscillation)
            if mode != "AVOID":
                turn_lbl, _ = _choose_turn_to_open_side(secs)
                step = math.radians(AVOID_TURN_STEP_DEG)
                target = _wrap_pi(cyaw + (step if turn_lbl == "TURN_LEFT" else -step))
                _nav_set("AVOID", target, reason=f"blocked->avoid({turn_lbl})")
                mode, target, set_ts = _nav_get()

        # Decide label based on heading error vs target
        err = _angle_err(target, cyaw)
        err_deg = math.degrees(err)
        deadband = float(HEADING_DEADBAND_DEG)

        if mode == "TRACK":
            # In TRACK, prefer GO_STRAIGHT, only correct if drift big
            if abs(err_deg) <= deadband:
                label = "GO_STRAIGHT"
                reason = "heading_lock_track"
            else:
                label = "TURN_LEFT" if err > 0 else "TURN_RIGHT"
                reason = "heading_lock_track_correct"
        else:
            # AVOID: keep turning until either clear OR we reached target then try GO_STRAIGHT (but only if not too blocked)
            avoid_age = now - set_ts
            if abs(err_deg) > deadband:
                label = "TURN_LEFT" if err > 0 else "TURN_RIGHT"
                reason = "heading_lock_avoid_turn_to_target"
            else:
                if (avoid_age >= float(MIN_AVOID_HOLD_S)) and (not block_go) and (center_min > float(STOP_NEAR_M)):
                    label = "GO_STRAIGHT"
                    reason = "heading_lock_avoid_forward"
                else:
                    # still blocked -> keep turning in the same direction as sign of last err
                    # if err ~ 0, choose open side NOW but do not reset target too frequently
                    turn_lbl, _ = _choose_turn_to_open_side(secs)
                    label = turn_lbl
                    reason = "heading_lock_avoid_keep_turn"

        # Safety override: if block_go and label==GO_STRAIGHT -> turn to open side
        if block_go and label == "GO_STRAIGHT":
            turn_lbl, turn_dbg = _choose_turn_to_open_side(secs)
            label = turn_lbl
            reason = f"front_arc_block_go -> {turn_lbl}"

        predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
        return {
            "ok": True,
            "label": label,
            "reason": reason,
            "ts": now,
            "debug": {
                "front_center_deg_used": float(front_abs),
                "back_deg_est": float(_last_back_deg) if _last_back_deg is not None else None,
                "sectors": {
                    "LEFT":   {"count": int(secs["LEFT"]["count"]),   "min_dist": float(secs["LEFT"]["min_dist"])},
                    "CENTER": {"count": int(secs["CENTER"]["count"]), "min_dist": float(secs["CENTER"]["min_dist"])},
                    "RIGHT":  {"count": int(secs["RIGHT"]["count"]),  "min_dist": float(secs["RIGHT"]["min_dist"])},
                },
                "predict_dist_m": float(predict_dist),
                "front_arc": {
                    "deg": float(FRONT_ARC_DEG),
                    "block_m": float(FRONT_ARC_BLOCK_M),
                    "hard_stop_m": float(FRONT_ARC_HARD_STOP_M),
                    "rearm_m": float(HARD_STOP_REARM_M),
                    "min_dist": float(fa_min),
                    "min_rel_deg": float(fa_rel) if fa_rel is not None else None,
                },
                "heading_lock": {
                    "enabled": True,
                    "mode": mode,
                    "target_yaw_deg": math.degrees(target) if target is not None else None,
                    "current_yaw_deg": math.degrees(cyaw),
                    "err_deg": float(err_deg),
                    "deadband_deg": float(deadband),
                    "front_clear": bool(front_clear),
                    "avoid_age_s": float(now - set_ts),
                },
                "stop_then_turn": {
                    "active": False,
                    "armed": _is_hard_stop_armed(),
                }
            }
        }

    # =========================
    # fallback (if heading lock disabled)
    # =========================
    turn_lbl, _ = _choose_turn_to_open_side(secs)
    if block_go:
        label = turn_lbl
        reason = "fallback_block_go"
    else:
        label = "GO_STRAIGHT"
        reason = "fallback_go"

    return {"ok": True, "label": label, "reason": reason, "ts": now}


# =======================
# Pose + Map update
# =======================
def _decision_to_twist(label: str) -> Tuple[float, float]:
    w = math.radians(TURN_RATE_DEG_S)
    if label == "GO_STRAIGHT":
        return (ROBOT_SPEED_MPS, 0.0)
    if label == "TURN_LEFT":
        return (0.0, +w)
    if label == "TURN_RIGHT":
        return (0.0, -w)
    if label == "BACK":
        return (-BACK_SPEED_MPS, 0.0)
    return (0.0, 0.0)

def _integrate_pose(dt: float, label: str):
    global pose_x, pose_y, pose_yaw
    v, w = _decision_to_twist(label)

    if abs(w) < 1e-6:
        dx = v * dt * math.cos(pose_yaw)
        dy = v * dt * math.sin(pose_yaw)
        pose_x += dx
        pose_y += dy
    else:
        pose_yaw = _wrap_pi(pose_yaw + w * dt)

def _world_to_grid(wx: float, wy: float) -> Optional[Tuple[int, int]]:
    ox = MAP_SIZE_M * 0.5
    oy = MAP_SIZE_M * 0.5
    gx = int((wx + ox) / MAP_RES_M)
    gy = int((wy + oy) / MAP_RES_M)
    if 0 <= gx < grid_w and 0 <= gy < grid_h:
        return gx, gy
    return None

def _grid_decay_and_hit(hit_cells: List[Tuple[int, int]]):
    for y in range(grid_h):
        row = grid[y]
        for x in range(grid_w):
            row[x] *= MAP_DECAY

    for (gx, gy) in hit_cells:
        v = grid[gy][gx] + MAP_HIT
        if v > MAP_MAX:
            v = MAP_MAX
        grid[gy][gx] = v

def _update_map_from_lidar(pts: List[Tuple[float, float, int, float, float, float]], front_abs: float):
    now = time.time()
    with _pose_lock:
        rx = pose_x
        ry = pose_y
        yaw = pose_yaw

    c = math.cos(yaw)
    s = math.sin(yaw)

    hit_cells: List[Tuple[int, int]] = []
    for (theta, dist_m, q, x, y, ts) in pts:
        if now - ts > RECENT_SEC:
            continue
        if dist_m <= 0.05 or dist_m > MAX_RANGE_M:
            continue

        rel = _rel_deg(theta, front_abs)
        rel_rad = math.radians(rel)

        x_f = dist_m * math.cos(rel_rad)
        y_l = dist_m * math.sin(rel_rad)

        wx = rx + (c * x_f - s * y_l)
        wy = ry + (s * x_f + c * y_l)

        cell = _world_to_grid(wx, wy)
        if cell:
            hit_cells.append(cell)

    with _grid_lock:
        _grid_decay_and_hit(hit_cells)

def map_worker():
    global _pose_last_ts
    while True:
        with decision_state_lock:
            lbl = latest_decision_label
            dec = dict(latest_decision_full)

        with lock:
            pts = list(latest_points)
            last_ts = float(latest_ts)

        now = time.time()
        if _pose_last_ts <= 0.0:
            _pose_last_ts = now

        dt = now - _pose_last_ts
        _pose_last_ts = now
        dt = max(0.0, min(0.25, dt))

        with _pose_lock:
            _integrate_pose(dt, lbl)

        if pts and (now - last_ts) < 2.0:
            front_abs = _get_front_center_deg()
            _update_map_from_lidar(pts, front_abs)

        try:
            png = _render_map_png(decision=dec)
            with cache_lock:
                _map_cache["ts"] = time.time()
                _map_cache["png"] = png
        except Exception:
            pass

        time.sleep(THROTTLE_S)


# =======================
# Rendering (PNG)
# =======================
def _render_map_png(decision: Dict[str, Any]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    with _pose_lock:
        rx, ry, yaw = pose_x, pose_y, pose_yaw

    with _grid_lock:
        g = [row[:] for row in grid]

    W = H = int(VIEW_SIZE_PX)
    img = Image.new("RGB", (W, H), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    ppm = (W * 0.5) / max(0.5, float(VIEW_RANGE_M))

    def world_to_px(wx: float, wy: float) -> Tuple[int, int]:
        dx = (wx - rx)
        dy = (wy - ry)
        px = int(W * 0.5 + dx * ppm)
        py = int(H * 0.5 - dy * ppm)
        return px, py

    view_min_x = rx - VIEW_RANGE_M
    view_max_x = rx + VIEW_RANGE_M
    view_min_y = ry - VIEW_RANGE_M
    view_max_y = ry + VIEW_RANGE_M

    ox = MAP_SIZE_M * 0.5
    oy = MAP_SIZE_M * 0.5
    gx0 = int((view_min_x + ox) / MAP_RES_M)
    gx1 = int((view_max_x + ox) / MAP_RES_M)
    gy0 = int((view_min_y + oy) / MAP_RES_M)
    gy1 = int((view_max_y + oy) / MAP_RES_M)
    gx0 = max(0, min(grid_w - 1, gx0))
    gx1 = max(0, min(grid_w - 1, gx1))
    gy0 = max(0, min(grid_h - 1, gy0))
    gy1 = max(0, min(grid_h - 1, gy1))

    for gy in range(gy0, gy1 + 1):
        row = g[gy]
        wy = (gy * MAP_RES_M) - oy
        for gx in range(gx0, gx1 + 1):
            v = row[gx]
            if v < 10.0:
                continue
            wx = (gx * MAP_RES_M) - ox
            px, py = world_to_px(wx, wy)
            if 0 <= px < W and 0 <= py < H:
                d = int(max(0, min(200, 220 - (v / MAP_MAX) * 200)))
                img.putpixel((px, py), (d, d, d))

    cx, cy = int(W * 0.5), int(H * 0.5)
    r = 8
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 20, 20))

    hx = cx + int(math.cos(yaw) * 25)
    hy = cy - int(math.sin(yaw) * 25)
    draw.line((cx, cy, hx, hy), fill=(0, 0, 0), width=3)

    sector_defs = {
        "RIGHT": (-90.0, -SECTOR_CENTER_DEG),
        "CENTER": (-SECTOR_CENTER_DEG, +SECTOR_CENTER_DEG),
        "LEFT": (+SECTOR_CENTER_DEG, +90.0),
    }

    label = str(decision.get("label", "STOP"))
    selected_sector = None
    if label == "GO_STRAIGHT":
        selected_sector = "CENTER"
    elif label == "TURN_LEFT":
        selected_sector = "LEFT"
    elif label == "TURN_RIGHT":
        selected_sector = "RIGHT"

    dbg = decision.get("debug", {}) if isinstance(decision, dict) else {}
    secdbg = dbg.get("sectors", {}) if isinstance(dbg, dict) else {}
    dL = float(secdbg.get("LEFT", {}).get("min_dist", float("inf"))) if isinstance(secdbg.get("LEFT", {}), dict) else float("inf")
    dC = float(secdbg.get("CENTER", {}).get("min_dist", float("inf"))) if isinstance(secdbg.get("CENTER", {}), dict) else float("inf")
    dR = float(secdbg.get("RIGHT", {}).get("min_dist", float("inf"))) if isinstance(secdbg.get("RIGHT", {}), dict) else float("inf")

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    ring_px = int(SECTOR_RING_M * ppm)
    bbox = (cx - ring_px, cy - ring_px, cx + ring_px, cy + ring_px)

    def rel_to_pil_deg(rel_deg: float) -> float:
        ang = yaw + math.radians(rel_deg)
        return math.degrees(-ang) % 360.0

    for name, (a0, a1) in sector_defs.items():
        start = rel_to_pil_deg(a1)
        end = rel_to_pil_deg(a0)
        if selected_sector == name:
            fill = (0, 255, 0, SECTOR_ALPHA)
        else:
            fill = (255, 0, 0, SECTOR_ALPHA)
        od.pieslice(bbox, start=start, end=end, fill=fill)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = None
        font_small = None

    def fmt_dist(d: float) -> str:
        if not math.isfinite(d):
            return "inf"
        return f"{d*100:.0f}cm"

    text_items = [
        ("LEFT",   +60.0, fmt_dist(dL)),
        ("CENTER",  0.0,  fmt_dist(dC)),
        ("RIGHT",  -60.0, fmt_dist(dR)),
    ]
    for name, rel, t in text_items:
        ang = yaw + math.radians(rel)
        tx = cx + int(math.cos(ang) * ring_px * 0.75)
        ty = cy - int(math.sin(ang) * ring_px * 0.75)
        draw.text((tx - 25, ty - 10), f"{name}:{t}", fill=(0, 0, 0), font=font_small)

    hud = [
        f"label: {label}",
        f"pose: x={rx:.2f} y={ry:.2f} yaw={math.degrees(yaw):.1f}deg",
        f"L={fmt_dist(dL)} C={fmt_dist(dC)} R={fmt_dist(dR)}",
        f"front_center_deg={_get_front_center_deg():.1f}  mirror={FRONT_MIRROR}",
    ]
    y0 = 8
    for s in hud:
        draw.text((8, y0), s, fill=(0, 0, 0), font=font)
        y0 += 22

    rr = int(VIEW_RANGE_M * ppm)
    draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=(150, 150, 150), width=2)

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
# Points payload
# =======================
def _front_angles_for_plot(theta_raw: float, front_center_abs: float) -> Dict[str, Any]:
    rel = _rel_deg(theta_raw, front_center_abs)
    angle_front_360 = _wrap_deg(rel)
    sec = _sector_name(rel)
    return {
        "rel_deg": float(rel),
        "angle_front_360": float(angle_front_360),
        "is_front_180": bool(abs(rel) <= FRONT_HALF_DEG),
        "k3_sector": sec,
    }

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

    with _pose_lock:
        rx, ry, yaw = pose_x, pose_y, pose_yaw

    st_active, st_next, st_stop_issued, st_reason = _stopturn_get_state()
    nav_mode, nav_target, nav_set_ts = _nav_get()

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
        "front_mirror": int(FRONT_MIRROR),

        "heading_lock": {
            "enabled": bool(HEADING_LOCK_ENABLE == 1),
            "mode": nav_mode,
            "target_yaw_deg": math.degrees(nav_target) if nav_target is not None else None,
            "age_s": float(now - nav_set_ts) if nav_set_ts else None,
            "deadband_deg": float(HEADING_DEADBAND_DEG),
            "avoid_step_deg": float(AVOID_TURN_STEP_DEG),
        },

        "stop_then_turn": {
            "active": bool(st_active),
            "stop_issued": bool(st_stop_issued),
            "next": st_next,
            "reason": st_reason,
            "hard_stop_armed": _is_hard_stop_armed(),
            "front_arc_deg": float(FRONT_ARC_DEG),
            "block_m": float(FRONT_ARC_BLOCK_M),
            "hard_stop_m": float(FRONT_ARC_HARD_STOP_M),
            "rearm_m": float(HARD_STOP_REARM_M),
        },

        "pose": {
            "x": rx, "y": ry, "yaw_deg": math.degrees(yaw)
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

@app.get("/api/map.png")
def api_map_png():
    with cache_lock:
        png = _map_cache["png"]
    if png is None:
        with decision_state_lock:
            dec = dict(latest_decision_full)
        png = _render_map_png(dec)
        with cache_lock:
            _map_cache["png"] = png
            _map_cache["ts"] = time.time()
    return Response(png, mimetype="image/png")

@app.get("/dashboard")
def dashboard():
    html = f"""
    <html>
      <head>
        <title>LiDAR 2D Map</title>
        <style>
          body {{ font-family: Arial; margin: 12px; }}
          .row {{ display:flex; gap:12px; align-items:flex-start; }}
          img {{ border:1px solid #ccc; border-radius:8px; }}
          .box {{ padding:10px; border:1px solid #ddd; border-radius:8px; min-width: 360px; }}
          .mono {{ font-family: monospace; white-space: pre; }}
        </style>
      </head>
      <body>
        <h3>LiDAR 2D Map (360°) + k=3 sector overlay</h3>
        <div class="row">
          <img id="map" src="/api/map.png?ts={time.time()}" width="{VIEW_SIZE_PX}" height="{VIEW_SIZE_PX}"/>
          <div class="box">
            <div><b>Decision</b></div>
            <div id="label" class="mono">loading...</div>
            <hr/>
            <div><b>Status</b></div>
            <div id="status" class="mono">loading...</div>
          </div>
        </div>

        <script>
          async function tick() {{
            try {{
              const d = await fetch('/api/decision').then(r=>r.json());
              const s = await fetch('/api/status').then(r=>r.json());
              document.getElementById('label').textContent =
                JSON.stringify({{
                  label: d.label,
                  reason: d.reason,
                  front_arc: (d.debug||{{}}).front_arc,
                  sectors: (d.debug||{{}}).sectors,
                  heading_lock: (d.debug||{{}}).heading_lock,
                  stop_then_turn: (d.debug||{{}}).stop_then_turn
                }}, null, 2);

              document.getElementById('status').textContent =
                JSON.stringify({{
                  running: s.running,
                  age_s: s.age_s,
                  pose: s.pose,
                  front_center: s.front_center_deg_used,
                  heading_lock: s.heading_lock,
                  stop_then_turn: s.stop_then_turn
                }}, null, 2);

              document.getElementById('map').src = '/api/map.png?ts=' + Date.now();
            }} catch(e) {{
              console.log(e);
            }}
          }}
          setInterval(tick, {THROTTLE_MS});
          tick();
        </script>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")

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

@app.post("/api/map_reset")
def api_map_reset():
    global pose_x, pose_y, pose_yaw, _pose_last_ts
    with _grid_lock:
        for y in range(grid_h):
            for x in range(grid_w):
                grid[y][x] = 0.0
    with _pose_lock:
        pose_x = pose_y = 0.0
        pose_yaw = 0.0
        _pose_last_ts = 0.0
    _stopturn_clear()
    _rearm_hard_stop()
    _nav_set("TRACK", 0.0, reason="reset")
    return jsonify({"ok": True})

@app.get("/")
def home():
    return Response(
        "<h3>lidarhub running</h3>"
        "<ul>"
        "<li><a href='/dashboard'>/dashboard</a></li>"
        "<li>/api/map.png</li>"
        "<li>/take_lidar_data</li>"
        "<li>/ask_lidar_decision</li>"
        "<li>/api/decision_label</li>"
        "<li>/api/decision</li>"
        "<li>/api/status</li>"
        "<li>/api/restart (POST)</li>"
        "<li>/api/map_reset (POST)</li>"
        "</ul>",
        mimetype="text/html"
    )


# =======================
# Main
# =======================
def main():
    _set_front_center_deg(FRONT_CENTER_DEG)
    _nav_set("TRACK", 0.0, reason="boot")
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    threading.Thread(target=points_worker, daemon=True).start()
    threading.Thread(target=decision_worker, daemon=True).start()
    threading.Thread(target=map_worker, daemon=True).start()
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
