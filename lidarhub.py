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

# Angle calibration (manual fallback)
FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))  # raw theta that points to robot front (if auto off)
FRONT_MIRROR = int(os.environ.get("FRONT_MIRROR", "0"))              # 1 = mirror left/right if needed

# Debug sectors (keep old keys for compatibility)
FRONT_WIDTH_DEG = float(os.environ.get("FRONT_WIDTH_DEG", "30.0"))
WIDE_WIDTH_DEG  = float(os.environ.get("WIDE_WIDTH_DEG", "70.0"))

# ===== Gap / pass check =====
ROBOT_WIDTH_M = float(os.environ.get("ROBOT_WIDTH_M", "0.15"))
CLEARANCE_MARGIN_M = float(os.environ.get("CLEARANCE_MARGIN_M", "0.03"))
LOOKAHEAD_M = float(os.environ.get("LOOKAHEAD_M", "1.20"))

# ===== Front-only window =====
FRONT_HALF_DEG = float(os.environ.get("FRONT_HALF_DEG", "90.0"))  # front 180deg = [-90..+90] relative

# ===== Clustering (auto-K heuristic) =====
CLUSTER_ANG_GAP_DEG = float(os.environ.get("CLUSTER_ANG_GAP_DEG", "2.5"))
CLUSTER_DIST_JUMP_M = float(os.environ.get("CLUSTER_DIST_JUMP_M", "0.25"))
CLUSTER_MIN_PTS     = int(os.environ.get("CLUSTER_MIN_PTS", "4"))
RECENT_SEC          = float(os.environ.get("RECENT_SEC", "0.7"))

# ===== Auto front calibration (fix "front/back reversed") =====
AUTO_FRONT_CALIB = int(os.environ.get("AUTO_FRONT_CALIB", "1"))  # 1=auto detect back wall then front=back+180
AUTO_CALIB_NEAR_M = float(os.environ.get("AUTO_CALIB_NEAR_M", "0.80"))  # consider near points as "wall"
AUTO_CALIB_BIN_DEG = float(os.environ.get("AUTO_CALIB_BIN_DEG", "10.0"))  # histogram bin size
AUTO_CALIB_SMOOTH = float(os.environ.get("AUTO_CALIB_SMOOTH", "0.25"))    # EMA smooth 0..1

# ===== Behavior: stop -> turn (TURN must be 2s) =====
STOP_HOLD_SEC = float(os.environ.get("STOP_HOLD_SEC", "0.35"))
TURN_HOLD_SEC = float(os.environ.get("TURN_HOLD_SEC", "2.0"))
TURN_HOLD_SEC = max(2.0, TURN_HOLD_SEC)

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

# Extra: obstacle state for new endpoint
obstacle_state_lock = threading.Lock()
latest_front_obstacles: Dict[str, Any] = {"ok": False, "reason": "init", "ts": 0.0}

# ===== Turn state machine (NO BACK) =====
_turn_lock = threading.Lock()
_phase: str = "NORMAL"     # NORMAL / STOPPING / TURNING
_phase_until: float = 0.0
_turn_label: str = ""      # TURN_LEFT / TURN_RIGHT

# ===== Dynamic front center estimate =====
_front_center_est_lock = threading.Lock()
_front_center_est_deg: float = FRONT_CENTER_DEG   # raw theta that corresponds to robot front (estimated)


# =======================
# Helpers
# =======================
def _wrap_deg(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def _wrap_rel_deg(a: float) -> float:
    """wrap to [-180, +180]"""
    a = _wrap_deg(a)
    if a > 180.0:
        a -= 360.0
    return a

def _ang_dist_deg(a: float, b: float) -> float:
    d = abs(_wrap_deg(a) - _wrap_deg(b))
    return min(d, 360.0 - d)

def _ema_angle_deg(prev_deg: float, new_deg: float, alpha: float) -> float:
    """
    EMA for circular angle:
      move prev toward new by shortest direction.
    """
    prev = _wrap_deg(prev_deg)
    new = _wrap_deg(new_deg)
    delta = _wrap_rel_deg(new - prev)  # [-180..180]
    out = prev + alpha * delta
    return _wrap_deg(out)

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
    recent_sec: float = RECENT_SEC
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

def _required_gap_width_m() -> float:
    return float(ROBOT_WIDTH_M + 2.0 * CLEARANCE_MARGIN_M)

def _get_front_center_deg() -> float:
    with _front_center_est_lock:
        return float(_front_center_est_deg)

def _set_front_center_deg(new_front_deg: float):
    global _front_center_est_deg
    with _front_center_est_lock:
        _front_center_est_deg = float(_wrap_deg(new_front_deg))

def _estimate_back_direction_from_near_points(
    pts: List[Tuple[float, float, int, float, float, float]],
    recent_sec: float
) -> Optional[float]:
    """
    Find direction where many near points exist (likely wall).
    Return raw theta (center of best bin) as "back direction".
    """
    now = time.time()
    bin_deg = max(2.0, float(AUTO_CALIB_BIN_DEG))
    bins = int(round(360.0 / bin_deg))
    if bins < 12:
        bins = 12

    counts = [0] * bins
    dist_sums = [0.0] * bins

    for (theta, dist_m, q, x, y, ts) in pts:
        if now - ts > recent_sec:
            continue
        if dist_m <= 0.02:
            continue
        if dist_m > float(AUTO_CALIB_NEAR_M):
            continue
        idx = int(_wrap_deg(theta) // bin_deg) % bins
        counts[idx] += 1
        dist_sums[idx] += float(dist_m)

    best_idx = None
    best_score = -1e9
    for i in range(bins):
        if counts[i] <= 0:
            continue
        mean_d = dist_sums[i] / max(1, counts[i])
        # score: many points + closer mean distance
        score = counts[i] - 3.0 * mean_d
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return None

    back_center = (best_idx + 0.5) * bin_deg
    return float(_wrap_deg(back_center))

def _auto_calibrate_front_center(
    pts: List[Tuple[float, float, int, float, float, float]],
    recent_sec: float
):
    """
    If AUTO_FRONT_CALIB=1:
      - detect back direction from near points
      - front = back + 180
      - smooth update to _front_center_est_deg
    """
    if int(AUTO_FRONT_CALIB) != 1:
        _set_front_center_deg(FRONT_CENTER_DEG)
        return

    back = _estimate_back_direction_from_near_points(pts, recent_sec=recent_sec)
    if back is None:
        # no strong wall found -> keep current estimate (or fallback)
        return

    front = _wrap_deg(back + 180.0)
    prev = _get_front_center_deg()
    alpha = float(max(0.05, min(1.0, AUTO_CALIB_SMOOTH)))
    smoothed = _ema_angle_deg(prev, front, alpha)
    _set_front_center_deg(smoothed)

def _rel_deg(theta_deg: float, center_deg: float) -> float:
    """
    relative angle in [-180, +180]
      rel > 0 => left (CCW) relative to center
    """
    rel = _wrap_rel_deg(theta_deg - center_deg)
    if int(FRONT_MIRROR) == 1:
        rel = -rel
    return rel

def _front_points_only(
    pts: List[Tuple[float, float, int, float, float, float]],
    recent_sec: float = RECENT_SEC
) -> List[Tuple[float, float, int, float, float, float, float]]:
    """
    Points in front 180deg relative to estimated front center:
      rel in [-FRONT_HALF_DEG, +FRONT_HALF_DEG]
    Output adds rel_deg at end:
      (theta, dist_m, q, x, y, ts, rel)
    """
    now = time.time()
    front_center = _get_front_center_deg()
    out = []
    for (theta, dist_m, q, x, y, ts) in pts:
        if now - ts > recent_sec:
            continue
        rel = _rel_deg(theta, front_center)
        if abs(rel) <= float(FRONT_HALF_DEG):
            out.append((theta, dist_m, q, x, y, ts, rel))
    out.sort(key=lambda t: t[6])  # rel ascending (-90..+90)
    return out

def _cluster_obstacles_auto_k(
    front_pts: List[Tuple[float, float, int, float, float, float, float]]
) -> List[Dict[str, Any]]:
    """
    Heuristic auto-K clustering:
      - sort by rel angle
      - new cluster when:
          Δangle > CLUSTER_ANG_GAP_DEG OR |Δdist| > CLUSTER_DIST_JUMP_M
      - drop clusters with < CLUSTER_MIN_PTS
    """
    if not front_pts:
        return []

    clusters: List[List[Tuple[float, float, float]]] = []  # (rel, dist, theta)
    cur: List[Tuple[float, float, float]] = []

    prev_rel = None
    prev_dist = None

    for (theta, dist_m, q, x, y, ts, rel) in front_pts:
        if prev_rel is None:
            cur = [(rel, dist_m, theta)]
            prev_rel = rel
            prev_dist = dist_m
            continue

        dang = abs(rel - prev_rel)
        ddist = abs(dist_m - (prev_dist if prev_dist is not None else dist_m))

        if (dang > float(CLUSTER_ANG_GAP_DEG)) or (ddist > float(CLUSTER_DIST_JUMP_M)):
            if cur:
                clusters.append(cur)
            cur = [(rel, dist_m, theta)]
        else:
            cur.append((rel, dist_m, theta))

        prev_rel = rel
        prev_dist = dist_m

    if cur:
        clusters.append(cur)

    obstacles: List[Dict[str, Any]] = []
    oid = 0
    for c in clusters:
        if len(c) < int(CLUSTER_MIN_PTS):
            continue
        rels = [p[0] for p in c]
        dists = [p[1] for p in c]
        thetas = [p[2] for p in c]
        obstacles.append({
            "id": oid,
            "n": len(c),
            "rel_min_deg": float(min(rels)),
            "rel_max_deg": float(max(rels)),
            "rel_mean_deg": float(sum(rels) / len(rels)),
            "theta_mean_deg": float(sum(thetas) / len(thetas)),
            "dist_min_m": float(min(dists)),
            "dist_mean_m": float(sum(dists) / len(dists)),
        })
        oid += 1

    obstacles.sort(key=lambda o: o["rel_mean_deg"])
    for i, o in enumerate(obstacles):
        o["id"] = i
    return obstacles

def _gap_width_m(gap_deg: float, pass_dist_m: float) -> float:
    """
    width = 2 * d * sin(gap/2)
    """
    if gap_deg <= 0 or pass_dist_m <= 0:
        return 0.0
    return 2.0 * float(pass_dist_m) * math.sin(math.radians(gap_deg) * 0.5)

def _rank_gaps_and_pick_safe_heading(obstacles: List[Dict[str, Any]]) -> Dict[str, Any]:
    req_w = _required_gap_width_m()
    right_bound = -float(FRONT_HALF_DEG)
    left_bound  = +float(FRONT_HALF_DEG)

    obs = list(obstacles)
    gaps: List[Dict[str, Any]] = []

    def pass_dist_between(d1: Optional[float], d2: Optional[float]) -> float:
        ds = []
        if d1 is not None and d1 > 0: ds.append(float(d1))
        if d2 is not None and d2 > 0: ds.append(float(d2))
        if not ds:
            return float(LOOKAHEAD_M)
        return float(min(float(LOOKAHEAD_M), max(0.25, min(ds))))

    if not obs:
        full_gap_deg = left_bound - right_bound  # 180
        pass_d = float(LOOKAHEAD_M)
        w = _gap_width_m(full_gap_deg, pass_d)
        safe = 0.0
        gap = {
            "type": "boundary",
            "from_deg": right_bound,
            "to_deg": left_bound,
            "gap_deg": float(full_gap_deg),
            "center_deg": float(safe),
            "pass_dist_m": float(pass_d),
            "width_m": float(w),
            "passable": bool(w >= req_w),
        }
        return {
            "ok": True,
            "k": 0,
            "obstacle_count": 0,
            "required_width_m": float(req_w),
            "safe_heading_deg": float(safe),
            "best_gap": gap,
            "gaps": [gap],
            "obstacles": [],
        }

    # right boundary -> first obstacle
    first = obs[0]
    g_from = right_bound
    g_to = float(first["rel_min_deg"])
    g_deg = max(0.0, g_to - g_from)
    pass_d = pass_dist_between(first.get("dist_min_m"), None)
    w = _gap_width_m(g_deg, pass_d)
    gaps.append({
        "type": "boundary_right",
        "from_deg": float(g_from),
        "to_deg": float(g_to),
        "gap_deg": float(g_deg),
        "center_deg": float((g_from + g_to) * 0.5),
        "pass_dist_m": float(pass_d),
        "width_m": float(w),
        "passable": bool(w >= req_w),
    })

    # between obstacles
    for i in range(len(obs) - 1):
        a = obs[i]
        b = obs[i + 1]
        g_from = float(a["rel_max_deg"])
        g_to = float(b["rel_min_deg"])
        g_deg = max(0.0, g_to - g_from)
        pass_d = pass_dist_between(a.get("dist_min_m"), b.get("dist_min_m"))
        w = _gap_width_m(g_deg, pass_d)
        gaps.append({
            "type": "between",
            "a_id": int(a["id"]),
            "b_id": int(b["id"]),
            "from_deg": float(g_from),
            "to_deg": float(g_to),
            "gap_deg": float(g_deg),
            "center_deg": float((g_from + g_to) * 0.5),
            "pass_dist_m": float(pass_d),
            "width_m": float(w),
            "passable": bool(w >= req_w),
        })

    # last obstacle -> left boundary
    last = obs[-1]
    g_from = float(last["rel_max_deg"])
    g_to = left_bound
    g_deg = max(0.0, g_to - g_from)
    pass_d = pass_dist_between(last.get("dist_min_m"), None)
    w = _gap_width_m(g_deg, pass_d)
    gaps.append({
        "type": "boundary_left",
        "from_deg": float(g_from),
        "to_deg": float(g_to),
        "gap_deg": float(g_deg),
        "center_deg": float((g_from + g_to) * 0.5),
        "pass_dist_m": float(pass_d),
        "width_m": float(w),
        "passable": bool(w >= req_w),
    })

    def score(g):
        wc = min(float(g.get("width_m", 0.0)), 2.0)
        steer_pen = 0.01 * abs(float(g.get("center_deg", 0.0)))
        return wc - steer_pen

    gaps_sorted = sorted(gaps, key=score, reverse=True)
    best = gaps_sorted[0] if gaps_sorted else None
    safe_heading = float(best["center_deg"]) if best else 0.0

    return {
        "ok": True,
        "k": len(obs),
        "obstacle_count": len(obs),
        "required_width_m": float(req_w),
        "safe_heading_deg": float(safe_heading),
        "best_gap": best,
        "gaps": gaps_sorted,
        "obstacles": obs,
    }

def _set_phase_stop_then_turn(now: float, turn_label: str):
    global _phase, _phase_until, _turn_label
    with _turn_lock:
        _phase = "STOPPING"
        _phase_until = now + float(STOP_HOLD_SEC)
        _turn_label = turn_label

def _set_phase_turn(now: float, turn_label: str):
    global _phase, _phase_until, _turn_label
    with _turn_lock:
        _phase = "TURNING"
        _phase_until = now + float(TURN_HOLD_SEC)
        _turn_label = turn_label

def _get_phase_label(now: float) -> Optional[Tuple[str, str, float]]:
    global _phase, _phase_until, _turn_label
    with _turn_lock:
        if _phase == "STOPPING":
            if now < _phase_until:
                return ("STOP", "phase:STOPPING", _phase_until)
            _phase = "TURNING"
            _phase_until = now + float(TURN_HOLD_SEC)
            return (_turn_label or "TURN_LEFT", "phase:TURNING(2s)", _phase_until)

        if _phase == "TURNING":
            if now < _phase_until:
                return (_turn_label or "TURN_LEFT", "phase:TURNING(2s)", _phase_until)
            _phase = "NORMAL"
            _phase_until = 0.0
            _turn_label = ""
            return None

        return None

def _turn_label_from_heading(h: float) -> str:
    return "TURN_LEFT" if float(h) > 0 else "TURN_RIGHT"


# =======================
# Decision (front-180 + clustering gaps + auto front calib)
# =======================
def _compute_decision_snapshot() -> Dict[str, Any]:
    global latest_front_obstacles, _phase, _phase_until, _turn_label

    with lock:
        pts = list(latest_points)
        last_ts = float(latest_ts)

    now = time.time()
    if (not pts) or ((now - last_ts) > 2.0):
        with _turn_lock:
            _phase = "NORMAL"
            _phase_until = 0.0
            _turn_label = ""

        with obstacle_state_lock:
            latest_front_obstacles = {"ok": False, "reason": "no_recent_lidar", "ts": now}

        return {"ok": False, "label": "STOP", "reason": "no_recent_lidar", "ts": now}

    # Auto calibrate front center (fix reversed front/back)
    _auto_calibrate_front_center(pts, recent_sec=RECENT_SEC)
    front_center = _get_front_center_deg()

    # obey phase first
    ph = _get_phase_label(now)

    # front points + clustering + gaps
    front_pts = _front_points_only(pts, recent_sec=RECENT_SEC)
    obstacles = _cluster_obstacles_auto_k(front_pts)
    gap_info = _rank_gaps_and_pick_safe_heading(obstacles)

    # update obstacle endpoint state
    with obstacle_state_lock:
        latest_front_obstacles = {
            "ok": True,
            "ts": now,
            "front_center_deg_used": float(front_center),
            "k": int(gap_info.get("k", 0)),
            "obstacle_count": int(gap_info.get("obstacle_count", 0)),
            "safe_heading_deg": float(gap_info.get("safe_heading_deg", 0.0)),
            "required_width_m": float(gap_info.get("required_width_m", _required_gap_width_m())),
            "best_gap": gap_info.get("best_gap", None),
            "note": "front=estimated (auto from back-wall) then use rel [-90..+90], cluster auto-K, pick best gap center as safe_heading_deg.",
        }

    predict_dist = (float(ROBOT_SPEED_MPS) * float(PREDICT_T_SEC)) + float(SAFETY_MARGIN_M)
    block_th = max(float(STOP_NEAR_M), float(predict_dist))

    # Debug distances pack (keep old keys)
    # Use estimated front_center for absolute debug sectors
    front_abs = _wrap_deg(front_center)
    left_abs  = _wrap_deg(front_abs + 90.0)
    right_abs = _wrap_deg(front_abs - 90.0)
    diag_l    = _wrap_deg(front_abs + 45.0)
    diag_r    = _wrap_deg(front_abs - 45.0)

    d_front_narrow = _min_dist_in_sector(pts, front_abs, FRONT_WIDTH_DEG)
    d_front_wide   = _min_dist_in_sector(pts, front_abs, WIDE_WIDTH_DEG)
    d_left_wide    = _min_dist_in_sector(pts, left_abs,  WIDE_WIDTH_DEG)
    d_right_wide   = _min_dist_in_sector(pts, right_abs, WIDE_WIDTH_DEG)
    d_diag_l       = _min_dist_in_sector(pts, diag_l, 35.0)
    d_diag_r       = _min_dist_in_sector(pts, diag_r, 35.0)

    dist_pack = {
        "front_narrow": d_front_narrow,
        "front_wide": d_front_wide,
        "left": d_left_wide,
        "right": d_right_wide,
        "diag_left": d_diag_l,
        "diag_right": d_diag_r,
    }

    thresh_pack = {
        "stop_near_m": float(STOP_NEAR_M),
        "predict_dist_m": float(predict_dist),
        "predict_t_sec": float(PREDICT_T_SEC),
        "robot_speed_mps": float(ROBOT_SPEED_MPS),
        "safety_margin_m": float(SAFETY_MARGIN_M),
        "block_th_m": float(block_th),
        "robot_width_m": float(ROBOT_WIDTH_M),
        "clearance_margin_m": float(CLEARANCE_MARGIN_M),
        "lookahead_m": float(LOOKAHEAD_M),
        "stop_hold_sec": float(STOP_HOLD_SEC),
        "turn_hold_sec": float(TURN_HOLD_SEC),
        "front_half_deg": float(FRONT_HALF_DEG),
        "cluster_ang_gap_deg": float(CLUSTER_ANG_GAP_DEG),
        "cluster_dist_jump_m": float(CLUSTER_DIST_JUMP_M),
        "cluster_min_pts": int(CLUSTER_MIN_PTS),
        "recent_sec": float(RECENT_SEC),
        "auto_front_calib": int(AUTO_FRONT_CALIB),
        "front_center_deg_used": float(front_center),
        "front_mirror": int(FRONT_MIRROR),
    }

    # If in phase, return phase label, keep debug
    if ph:
        lbl, rs, until = ph
        thresh_pack["phase_until"] = float(until)
        safe_h = float(gap_info.get("safe_heading_deg", 0.0))
        best_gap = (gap_info.get("best_gap") or {})
        best_w = float(best_gap.get("width_m", 0.0) or 0.0)
        return {
            "ok": True,
            "label": str(lbl),
            "reason": str(rs),
            "ts": now,
            "dist": dist_pack,
            "threshold": thresh_pack,
            "corridor": {  # keep key name for compatibility
                "best_heading_deg": float(safe_h),
                "best_clearance_m": float(best_w),
                "method": "gap_cluster_front180_auto_front",
            },
            "cluster": gap_info,
        }

    # ===== normal evaluation =====
    safe_h = float(gap_info.get("safe_heading_deg", 0.0))
    best_gap = (gap_info.get("best_gap") or {})
    best_w = float(best_gap.get("width_m", 0.0) or 0.0)
    passable = bool(best_gap.get("passable", False))

    # nearest in front
    nearest_front = float("inf")
    for (_, dist_m, *_rest) in front_pts:
        if dist_m > 0 and dist_m < nearest_front:
            nearest_front = float(dist_m)

    # too near => STOP then TURN
    if nearest_front <= float(STOP_NEAR_M):
        if passable:
            turn_lbl = _turn_label_from_heading(safe_h)
        else:
            turn_lbl = "TURN_LEFT" if d_left_wide >= d_right_wide else "TURN_RIGHT"
        _set_phase_stop_then_turn(now, turn_lbl)
        return {
            "ok": True,
            "label": "STOP",
            "reason": "very_near_stop_then_turn(gap_cluster_auto_front)",
            "ts": now,
            "dist": dist_pack,
            "threshold": thresh_pack,
            "corridor": {
                "best_heading_deg": float(safe_h),
                "best_clearance_m": float(best_w),
                "method": "gap_cluster_front180_auto_front",
            },
            "cluster": gap_info,
        }

    # passable gap
    if passable:
        # small steer => go straight
        if abs(safe_h) <= 12.0:
            return {
                "ok": True,
                "label": "GO_STRAIGHT",
                "reason": "safe_gap_center_near_forward",
                "ts": now,
                "dist": dist_pack,
                "threshold": thresh_pack,
                "corridor": {
                    "best_heading_deg": float(safe_h),
                    "best_clearance_m": float(best_w),
                    "method": "gap_cluster_front180_auto_front",
                },
                "cluster": gap_info,
            }

        # turn 2s then re-check
        lbl = _turn_label_from_heading(safe_h)
        _set_phase_turn(now, lbl)
        return {
            "ok": True,
            "label": lbl,
            "reason": "turn_to_safe_gap_hold_2s(gap_cluster_auto_front)",
            "ts": now,
            "dist": dist_pack,
            "threshold": thresh_pack,
            "corridor": {
                "best_heading_deg": float(safe_h),
                "best_clearance_m": float(best_w),
                "method": "gap_cluster_front180_auto_front",
            },
            "cluster": gap_info,
        }

    # no passable gap -> stop then search
    gaps = (gap_info.get("gaps") or [])
    best_any = gaps[0] if gaps else None
    if best_any:
        lbl = _turn_label_from_heading(float(best_any.get("center_deg", 0.0)))
    else:
        lbl = "TURN_LEFT" if d_left_wide >= d_right_wide else "TURN_RIGHT"

    _set_phase_stop_then_turn(now, lbl)
    return {
        "ok": True,
        "label": "STOP",
        "reason": "no_passable_gap_stop_then_turn_search(gap_cluster_auto_front)",
        "ts": now,
        "dist": dist_pack,
        "threshold": thresh_pack,
        "corridor": {
            "best_heading_deg": float(safe_h),
            "best_clearance_m": float(best_w),
            "method": "gap_cluster_front180_auto_front",
        },
        "cluster": gap_info,
    }


# =======================
# Points payload (compat)
# =======================
def _build_points_payload(limit: int = 1600) -> Dict[str, Any]:
    with lock:
        pts = list(latest_points[-limit:])
        ts = float(latest_ts)

    out = []
    for (theta, dist_m, q, x, y, t) in pts:
        out.append({
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
            "front_center_deg": float(_get_front_center_deg()),  # now dynamic (auto)
            "max_range_m": float(MAX_RANGE_M),
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
        ts = float(latest_ts)

    with decision_state_lock:
        lbl = latest_decision_label

    with _turn_lock:
        phase = {
            "phase": _phase,
            "phase_until": _phase_until,
            "turn_label": _turn_label
        }

    with obstacle_state_lock:
        ob = dict(latest_front_obstacles)

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
        "robot_width_m": float(ROBOT_WIDTH_M),
        "required_gap_width_m": float(_required_gap_width_m()),
        "lookahead_m": float(LOOKAHEAD_M),
        "front_half_deg": float(FRONT_HALF_DEG),
        "front_center_deg_used": float(_get_front_center_deg()),
        "auto_front_calib": int(AUTO_FRONT_CALIB),
        "front_mirror": int(FRONT_MIRROR),
        "phase_state": phase,
        "front_obstacles": ob,
        "note": "Front direction is auto-calibrated (back wall -> front=back+180) to fix reversed front/back. Endpoints kept compatible.",
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

# NEW endpoint: obstacles + K + safe angle
@app.get("/api/front_obstacles")
def api_front_obstacles():
    with obstacle_state_lock:
        payload = dict(latest_front_obstacles)
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
        "<li>/api/front_obstacles (NEW)</li>"
        "<li>/api/restart (POST)</li>"
        "</ul>",
        mimetype="text/html"
    )

def main():
    # initialize estimated front (manual or auto)
    _set_front_center_deg(FRONT_CENTER_DEG)

    threading.Thread(target=lidar_thread_main, daemon=True).start()
    threading.Thread(target=points_worker, daemon=True).start()
    threading.Thread(target=decision_worker, daemon=True).start()

    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
