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
STOP_NEAR_M = float(os.environ.get("STOP_NEAR_M", "0.30"))      # stop if center < 30cm
LOOKAHEAD_M = float(os.environ.get("LOOKAHEAD_M", "1.20"))      # obstacle counting range for sector stats
PREDICT_T_SEC = float(os.environ.get("PREDICT_T_SEC", "1.0"))
ROBOT_SPEED_MPS = float(os.environ.get("ROBOT_SPEED_MPS", "0.35"))
SAFETY_MARGIN_M = float(os.environ.get("SAFETY_MARGIN_M", "0.10"))

# Plot/Heading
FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))
FRONT_MIRROR = int(os.environ.get("FRONT_MIRROR", "0"))  # set 1 if left/right inverted

RECENT_SEC = float(os.environ.get("RECENT_SEC", "0.7"))

# Auto front calib (back wall -> front = back + 180)
AUTO_FRONT_CALIB = int(os.environ.get("AUTO_FRONT_CALIB", "1"))
AUTO_CALIB_BIN_DEG = float(os.environ.get("AUTO_CALIB_BIN_DEG", "10.0"))
AUTO_CALIB_SMOOTH = float(os.environ.get("AUTO_CALIB_SMOOTH", "0.35"))
AUTO_CALIB_NEAR_CAP_M = float(os.environ.get("AUTO_CALIB_NEAR_CAP_M", "2.0"))
AUTO_CALIB_PERCENTILE = float(os.environ.get("AUTO_CALIB_PERCENTILE", "20.0"))

# k=3 sectors in FRONT 180
SECTOR_CENTER_DEG = float(os.environ.get("SECTOR_CENTER_DEG", "30.0"))  # center half-width (default +-30)
FRONT_HALF_DEG = float(os.environ.get("FRONT_HALF_DEG", "90.0"))        # front 180 = [-90..+90]

# Sticky turn
CLEAR_RELEASE_M = float(os.environ.get("CLEAR_RELEASE_M", "0.40"))      # release when center_min > 40cm
CLEAR_CONFIRM_SEC = float(os.environ.get("CLEAR_CONFIRM_SEC", "0.25"))  # require stable clear a bit
TURN_STICKY_MIN_SEC = float(os.environ.get("TURN_STICKY_MIN_SEC", "0.35"))

# ===== NEW: Front-arc safety (Fix GO_STRAIGHT) =====
# Nếu trong front arc (±FRONT_ARC_DEG) có điểm gần hơn FRONT_ARC_BLOCK_M => chặn GO_STRAIGHT (TURN/STOP)
FRONT_ARC_DEG = float(os.environ.get("FRONT_ARC_DEG", "85.0"))
FRONT_ARC_BLOCK_M = float(os.environ.get("FRONT_ARC_BLOCK_M", "0.55"))
FRONT_ARC_HARD_STOP_M = float(os.environ.get("FRONT_ARC_HARD_STOP_M", "0.35"))

# STOP rồi mới TURN (để robot dừng lại 1 nhịp)
STOP_THEN_TURN_HOLD_SEC = float(os.environ.get("STOP_THEN_TURN_HOLD_SEC", "0.20"))

# ===== Map (occupancy) =====
MAP_SIZE_M = float(os.environ.get("MAP_SIZE_M", "10.0"))   # world map width/height in meters
MAP_RES_M = float(os.environ.get("MAP_RES_M", "0.05"))     # meters per cell (0.05 => 20 cells/m)
MAP_DECAY = float(os.environ.get("MAP_DECAY", "0.985"))    # 0.985 => fade old points slowly
MAP_HIT = float(os.environ.get("MAP_HIT", "30.0"))         # add value per hit
MAP_MAX = float(os.environ.get("MAP_MAX", "255.0"))

# pose dead-reckoning (based on decision)
TURN_RATE_DEG_S = float(os.environ.get("TURN_RATE_DEG_S", "70.0"))  # turning angular speed estimate
BACK_SPEED_MPS = float(os.environ.get("BACK_SPEED_MPS", "0.20"))    # if label BACK exists later

# view render
VIEW_SIZE_PX = int(os.environ.get("VIEW_SIZE_PX", "640"))           # output image px
VIEW_RANGE_M = float(os.environ.get("VIEW_RANGE_M", "3.5"))         # visible radius around robot
SECTOR_RING_M = float(os.environ.get("SECTOR_RING_M", "1.2"))        # ring overlay radius
SECTOR_ALPHA = int(os.environ.get("SECTOR_ALPHA", "90"))             # 0..255

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

# Sticky turn state
_sticky_lock = threading.Lock()
_sticky_label: Optional[str] = None            # "TURN_LEFT" / "TURN_RIGHT" / None
_sticky_since: float = 0.0
_sticky_clear_start: Optional[float] = None

# ===== NEW: STOP-then-TURN state (keeps endpoints unchanged) =====
_stop_turn_lock = threading.Lock()
_stop_hold_until: float = 0.0
_pending_turn_label: Optional[str] = None
_pending_turn_reason: str = ""

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
grid = [[0.0 for _ in range(grid_w)] for __ in range(grid_h)]  # float for decay


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
    # only use front half for k=3 decision
    if abs(rel_deg) > FRONT_HALF_DEG:
        return None
    c = float(SECTOR_CENTER_DEG)
    if rel_deg < -c:
        return "RIGHT"
    if rel_deg > c:
        return "LEFT"
    return "CENTER"


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

def _choose_direction_from_sectors(secs: Dict[str, Dict[str, float]]) -> Tuple[str, str, Dict[str, Any]]:
    cmin = float(secs["CENTER"]["min_dist"])
    if cmin <= STOP_NEAR_M:
        return ("STOP", "center_too_close", {"center_min": cmin, "stop_near_m": STOP_NEAR_M})

    lc = secs["LEFT"]["count"]
    cc = secs["CENTER"]["count"]
    rc = secs["RIGHT"]["count"]

    m = min(lc, cc, rc)
    candidates = []
    if lc == m: candidates.append("LEFT")
    if cc == m: candidates.append("CENTER")
    if rc == m: candidates.append("RIGHT")

    if "CENTER" in candidates:
        best = "CENTER"
    else:
        lmin = float(secs["LEFT"]["min_dist"])
        rmin = float(secs["RIGHT"]["min_dist"])
        best = "LEFT" if lmin >= rmin else "RIGHT"

    if best == "CENTER":
        return ("GO_STRAIGHT", "k3_best_center", {"counts": {"L": lc, "C": cc, "R": rc}})
    if best == "LEFT":
        return ("TURN_LEFT", "k3_best_left", {"counts": {"L": lc, "C": cc, "R": rc}})
    return ("TURN_RIGHT", "k3_best_right", {"counts": {"L": lc, "C": cc, "R": rc}})

def _apply_sticky_turn(desired_label: str, center_min: float, now: float) -> Tuple[str, str, Dict[str, Any]]:
    global _sticky_label, _sticky_since, _sticky_clear_start

    with _sticky_lock:
        cur = _sticky_label

        if cur in ("TURN_LEFT", "TURN_RIGHT"):
            held_for = now - _sticky_since

            if held_for < TURN_STICKY_MIN_SEC:
                return (cur, "sticky_min_hold", {"held_for": held_for})

            if center_min > CLEAR_RELEASE_M:
                if _sticky_clear_start is None:
                    _sticky_clear_start = now
                if (now - _sticky_clear_start) >= CLEAR_CONFIRM_SEC:
                    _sticky_label = None
                    _sticky_clear_start = None
                    cur = None
                else:
                    return (cur, "sticky_wait_clear_confirm", {"center_min": center_min, "clear_for": now - _sticky_clear_start})
            else:
                _sticky_clear_start = None
                return (cur, "sticky_turn_until_front_clear", {"center_min": center_min, "clear_release_m": CLEAR_RELEASE_M})

        if cur is None and desired_label in ("TURN_LEFT", "TURN_RIGHT"):
            _sticky_label = desired_label
            _sticky_since = now
            _sticky_clear_start = None
            return (desired_label, "set_sticky_turn", {"center_min": center_min})

    return (desired_label, "not_sticky", {"center_min": center_min})

def _clear_sticky_now():
    global _sticky_label, _sticky_since, _sticky_clear_start
    with _sticky_lock:
        _sticky_label = None
        _sticky_since = 0.0
        _sticky_clear_start = None

def _choose_turn_to_open_side(secs: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, Any]]:
    """
    Chọn hướng TURN theo phía trống hơn.
    Ưu tiên min_dist lớn hơn; nếu bằng nhau thì ưu tiên count ít hơn.
    """
    lmin = float(secs["LEFT"]["min_dist"])
    rmin = float(secs["RIGHT"]["min_dist"])
    lc = float(secs["LEFT"]["count"])
    rc = float(secs["RIGHT"]["count"])

    # min_dist thắng
    if math.isfinite(lmin) and math.isfinite(rmin):
        if lmin > rmin + 1e-6:
            return "TURN_LEFT", {"pick": "min_dist", "lmin": lmin, "rmin": rmin, "lc": lc, "rc": rc}
        if rmin > lmin + 1e-6:
            return "TURN_RIGHT", {"pick": "min_dist", "lmin": lmin, "rmin": rmin, "lc": lc, "rc": rc}

    # fallback: count ít hơn
    if lc <= rc:
        return "TURN_LEFT", {"pick": "count", "lmin": lmin, "rmin": rmin, "lc": lc, "rc": rc}
    return "TURN_RIGHT", {"pick": "count", "lmin": lmin, "rmin": rmin, "lc": lc, "rc": rc}

def _front_arc_min(
    pts: List[Tuple[float, float, int, float, float, float]],
    front_abs: float
) -> Tuple[float, Optional[float]]:
    """
    Return (min_dist, min_rel_deg) in front arc ±FRONT_ARC_DEG within RECENT_SEC.
    """
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

def _set_stop_then_turn(now: float, turn_label: str, reason: str):
    global _stop_hold_until, _pending_turn_label, _pending_turn_reason
    with _stop_turn_lock:
        _stop_hold_until = now + float(STOP_THEN_TURN_HOLD_SEC)
        _pending_turn_label = turn_label
        _pending_turn_reason = reason

def _consume_pending_turn_if_ready(now: float) -> Tuple[bool, Optional[str], str]:
    """
    Nếu đang ở chế độ STOP->TURN và đã hết hold time => trả (True, label, reason) và clear pending.
    """
    global _stop_hold_until, _pending_turn_label, _pending_turn_reason
    with _stop_turn_lock:
        if _pending_turn_label and now >= _stop_hold_until:
            lbl = _pending_turn_label
            rs = _pending_turn_reason
            _pending_turn_label = None
            _pending_turn_reason = ""
            _stop_hold_until = 0.0
            return True, lbl, rs
    return False, None, ""

def _is_in_stop_hold(now: float) -> bool:
    with _stop_turn_lock:
        return (_pending_turn_label is not None) and (now < _stop_hold_until)


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

    # ===== NEW: front arc check (fix GO_STRAIGHT) =====
    fa_min, fa_rel = _front_arc_min(pts, front_abs)

    # ===== NEW: STOP->TURN state machine =====
    # Nếu đang hold STOP thì luôn STOP (không cho sticky override)
    if _is_in_stop_hold(now):
        predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
        return {
            "ok": True,
            "label": "STOP",
            "reason": "stop_then_turn_hold",
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
                    "min_dist": float(fa_min),
                    "min_rel_deg": float(fa_rel) if fa_rel is not None else None,
                    "block_m": float(FRONT_ARC_BLOCK_M),
                    "hard_stop_m": float(FRONT_ARC_HARD_STOP_M),
                },
            }
        }

    # Nếu hết hold => phải TURN theo hướng trống (label này sẽ đi qua sticky để giữ turn)
    ready, pending_lbl, pending_reason = _consume_pending_turn_if_ready(now)
    if ready and pending_lbl in ("TURN_LEFT", "TURN_RIGHT"):
        desired_label = pending_lbl
        desired_reason = f"post_stop_turn | {pending_reason}"
        desired_dbg = {"post_stop_turn": True, "pending": pending_lbl}
    else:
        desired_label, desired_reason, desired_dbg = _choose_direction_from_sectors(secs)

    # ===== NEW: Apply front-arc rule to BLOCK GO_STRAIGHT =====
    # Case A: Hard stop => STOP first, then schedule TURN next tick
    hard_stop = (fa_min <= float(FRONT_ARC_HARD_STOP_M))
    block_go = (fa_min <= float(FRONT_ARC_BLOCK_M))

    if hard_stop:
        # chọn hướng turn trống hơn rồi schedule
        turn_lbl, turn_dbg = _choose_turn_to_open_side(secs)
        _set_stop_then_turn(now, turn_lbl, reason=f"front_arc_hard_stop(min={fa_min:.3f}, rel={fa_rel}) -> {turn_lbl}")
        # hard stop override sticky (an toàn)
        _clear_sticky_now()

        predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M
        return {
            "ok": True,
            "label": "STOP",
            "reason": "front_arc_hard_stop_then_turn",
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
                    "min_dist": float(fa_min),
                    "min_rel_deg": float(fa_rel) if fa_rel is not None else None,
                    "block_m": float(FRONT_ARC_BLOCK_M),
                    "hard_stop_m": float(FRONT_ARC_HARD_STOP_M),
                },
                "stop_then_turn": {
                    "hold_sec": float(STOP_THEN_TURN_HOLD_SEC),
                    "next_turn": turn_lbl,
                    "turn_debug": turn_dbg,
                }
            }
        }

    # Case B: block GO_STRAIGHT => nếu desired là GO_STRAIGHT thì đổi sang TURN theo hướng trống
    if block_go and desired_label == "GO_STRAIGHT":
        turn_lbl, turn_dbg = _choose_turn_to_open_side(secs)
        desired_label = turn_lbl
        desired_reason = f"front_arc_block_go_straight(min={fa_min:.3f}, rel={fa_rel}) -> {turn_lbl}"
        desired_dbg = {**desired_dbg, "front_arc_override": True, "turn_debug": turn_dbg}

    # Sticky (GIỮ NGUYÊN)
    final_label, sticky_reason, sticky_dbg = _apply_sticky_turn(desired_label, center_min, now)

    predict_dist = (ROBOT_SPEED_MPS * PREDICT_T_SEC) + SAFETY_MARGIN_M

    return {
        "ok": True,
        "label": final_label,
        "reason": f"{desired_reason} | {sticky_reason}",
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
                "min_dist": float(fa_min),
                "min_rel_deg": float(fa_rel) if fa_rel is not None else None,
                "block_m": float(FRONT_ARC_BLOCK_M),
                "hard_stop_m": float(FRONT_ARC_HARD_STOP_M),
            },
            "desired": {"label": desired_label, "reason": desired_reason, **(desired_dbg or {})},
            "sticky": {"label": final_label, "reason": sticky_reason, **(sticky_dbg or {})},
        }
    }


# =======================
# Pose + Map update
# =======================
def _decision_to_twist(label: str) -> Tuple[float, float]:
    """
    returns (v, w) in robot local frame:
      v: m/s forward
      w: rad/s positive = turn left
    """
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
        pose_yaw += w * dt

    while pose_yaw > math.pi:
        pose_yaw -= 2 * math.pi
    while pose_yaw < -math.pi:
        pose_yaw += 2 * math.pi

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
            rx, ry, yaw = pose_x, pose_y, pose_yaw

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
        end   = rel_to_pil_deg(a0)

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
# Points payload (360° + extra info)
# =======================
def _front_angles_for_plot(theta_raw: float, front_center_abs: float) -> Dict[str, Any]:
    rel = _rel_deg(theta_raw, front_center_abs)
    angle_front_360 = _wrap_deg(rel)  # 0..360 where 0=front
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

    with _sticky_lock:
        sticky = _sticky_label
        sticky_since = _sticky_since
        clear_start = _sticky_clear_start

    with _pose_lock:
        rx, ry, yaw = pose_x, pose_y, pose_yaw

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

        "pose": {
            "x": rx, "y": ry, "yaw_deg": math.degrees(yaw)
        },

        "front_arc_rule": {
            "deg": float(FRONT_ARC_DEG),
            "block_m": float(FRONT_ARC_BLOCK_M),
            "hard_stop_m": float(FRONT_ARC_HARD_STOP_M),
            "stop_then_turn_hold_sec": float(STOP_THEN_TURN_HOLD_SEC),
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
          .box {{ padding:10px; border:1px solid #ddd; border-radius:8px; min-width: 320px; }}
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
                JSON.stringify({{label: d.label, reason: d.reason, sectors: (d.debug||{{}}).sectors, front_arc: (d.debug||{{}}).front_arc}}, null, 2);
              document.getElementById('status').textContent =
                JSON.stringify({{running: s.running, age_s: s.age_s, pose: s.pose, front_center: s.front_center_deg_used}}, null, 2);

              const img = document.getElementById('map');
              img.src = '/api/map.png?ts=' + Date.now();
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
    global grid, pose_x, pose_y, pose_yaw, _pose_last_ts
    with _grid_lock:
        for y in range(grid_h):
            for x in range(grid_w):
                grid[y][x] = 0.0
    with _pose_lock:
        pose_x = pose_y = 0.0
        pose_yaw = 0.0
        _pose_last_ts = 0.0
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
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    threading.Thread(target=points_worker, daemon=True).start()
    threading.Thread(target=decision_worker, daemon=True).start()
    threading.Thread(target=map_worker, daemon=True).start()
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
