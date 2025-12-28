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

# ============================================================
# ORIENTATION LOCKED (đúng theo hướng mặt robot của bạn)
# ============================================================
FRONT_CENTER_DEG_HARD = 0.0   # set cứng
FRONT_MIRROR_HARD = 0         # set cứng
FRONT_FLIP_HARD = 1           # set cứng (đảo front/back)
LOCK_ORIENTATION = True

# =======================
# CONFIG
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
POINT_LIMIT = int(os.environ.get("POINT_LIMIT", "4000"))

THROTTLE_MS = int(os.environ.get("THROTTLE_MS", "120"))
THROTTLE_S = THROTTLE_MS / 1000.0

# ===== FRONT 180 =====
FRONT_HALF_DEG = float(os.environ.get("FRONT_HALF_DEG", "90.0"))

# Snapshot window: giảm smear khi robot đang xoay
FRAME_SEC = float(os.environ.get("FRAME_SEC", "0.20"))

# ===== k=3 sectors =====
SECTOR_CENTER_DEG = float(os.environ.get("SECTOR_CENTER_DEG", "30.0"))
K_NEAR = int(os.environ.get("K_NEAR", "6"))
MIN_SECTOR_POINTS = int(os.environ.get("MIN_SECTOR_POINTS", "20"))

# ===== Thresholds =====
CLEAR_GO_M = float(os.environ.get("CLEAR_GO_M", "0.95"))            # clear để đi thẳng (hành lang)
OBSTACLE_NEAR_M = float(os.environ.get("OBSTACLE_NEAR_M", "0.70"))  # nếu bất kỳ vùng nào quá gần => STOP rồi TURN
CLEAR_STREAK_N = int(os.environ.get("CLEAR_STREAK_N", "2"))

# Emergency STOP one-shot (cực gần)
EMERGENCY_STOP_M = float(os.environ.get("EMERGENCY_STOP_M", "0.25"))
EMERGENCY_REARM_M = float(os.environ.get("EMERGENCY_REARM_M", "0.45"))

LABEL_CONFIRM_N = int(os.environ.get("LABEL_CONFIRM_N", "2"))

# ===== Avoid timeout (tránh bị kẹt turn vô hạn) =====
AVOID_MAX_SEC = float(os.environ.get("AVOID_MAX_SEC", "5.0"))

# ===== Gap Planner / Corridor =====
# >>> UPDATE: Robot rộng 40cm (theo bạn) <<<
ROBOT_WIDTH_M = float(os.environ.get("ROBOT_WIDTH_M", "0.40"))  # 40cm

# Margin để không cạ (giữ như bạn; nếu muốn "vừa khít" hơn thì có thể giảm env xuống 0.02~0.03)
CORRIDOR_MARGIN_M = float(os.environ.get("CORRIDOR_MARGIN_M", "0.04"))

# Extra safety (nếu muốn cho "lọt vừa", set = 0.0 hoặc 0.01)
GAP_EXTRA_M = float(os.environ.get("GAP_EXTRA_M", "0.00"))

LOOKAHEAD_M = float(os.environ.get("LOOKAHEAD_M", "0.80"))

BIN_DEG = float(os.environ.get("BIN_DEG", "5.0"))  # 5 độ/bin

# (giữ lại như "floor" tối thiểu theo độ để tránh gap cực nhỏ do nhiễu)
GAP_MIN_WIDTH_DEG = float(os.environ.get("GAP_MIN_WIDTH_DEG", "10.0"))

TURN_DEADBAND_DEG = float(os.environ.get("TURN_DEADBAND_DEG", "12.0"))

# chống lắc trái/phải
AVOID_COMMIT_SEC = float(os.environ.get("AVOID_COMMIT_SEC", "0.9"))     # giữ hướng turn tối thiểu
AVOID_SWITCH_HYST = float(os.environ.get("AVOID_SWITCH_HYST", "0.18"))  # chỉ đổi khi tốt hơn rõ rệt

# ===== Rendering =====
VIEW_SIZE_PX = int(os.environ.get("VIEW_SIZE_PX", "720"))
VIEW_RANGE_M = float(os.environ.get("VIEW_RANGE_M", "3.5"))
SECTOR_RING_M = float(os.environ.get("SECTOR_RING_M", "1.25"))
SECTOR_ALPHA = int(os.environ.get("SECTOR_ALPHA", "90"))

LINE_RE = re.compile(r"theta:\s*([0-9.]+)\s+Dist:\s*([0-9.]+)\s+Q:\s*(\d+)")

app = Flask(__name__)

# =======================
# Shared state
# =======================
lock = threading.Lock()
latest_points: List[Tuple[float, float, int, float]] = []  # (theta_deg, dist_m, q, ts)
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
latest_decision_full: Dict[str, Any] = {"ok": False, "label": "STOP", "reason": "init", "ts": 0.0}
latest_decision_label: str = "STOP"

# ===== Avoid state (bật sau STOP) =====
_avoid_lock = threading.Lock()
_avoid_active: bool = False
_avoid_clear_streak: int = 0
_avoid_dir: str = "TURN_RIGHT"  # TURN_RIGHT / TURN_LEFT
_avoid_start_ts: float = 0.0
_avoid_last_switch_ts: float = 0.0

# ===== Emergency gate =====
_em_lock = threading.Lock()
_em_stop_armed: bool = True
_em_stop_issued_this_event: bool = False

# ===== Label debounce =====
_label_lock = threading.Lock()
_out_label: str = "STOP"
_pending_label: str = ""
_pending_n: int = 0


# =======================
# Geometry helpers for "khe hẹp"
# =======================
def _required_clear_width_m() -> float:
    # khoảng trống cần để robot chui qua (vừa thân + 2 bên margin + extra)
    return float(ROBOT_WIDTH_M) + 2.0 * float(CORRIDOR_MARGIN_M) + float(GAP_EXTRA_M)

def _gap_width_m(width_deg: float, dist_m: float) -> float:
    # width (m) của khe ở khoảng cách dist_m
    if not math.isfinite(dist_m) or dist_m <= 0:
        return 0.0
    a = math.radians(max(0.0, float(width_deg)) * 0.5)
    return 2.0 * float(dist_m) * math.tan(a)

def _min_gap_width_deg(dist_m: float) -> float:
    # minimum degrees để đạt required width tại dist_m
    req = _required_clear_width_m()
    if not math.isfinite(dist_m) or dist_m <= 0:
        return 180.0
    # avoid zero division
    x = req / (2.0 * float(dist_m))
    if x <= 0:
        return 0.0
    # clamp x to avoid atan inf weirdness (still fine if large -> near 180)
    a = math.degrees(2.0 * math.atan(x))
    return float(max(0.0, min(180.0, a)))


# =======================
# Orientation helpers
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

def _rel_deg(theta_deg: float) -> float:
    # locked orientation
    fc = float(FRONT_CENTER_DEG_HARD)
    if int(FRONT_FLIP_HARD) == 1:
        fc = _wrap_deg(fc + 180.0)
    rel = _wrap_rel_deg(theta_deg - fc)
    if int(FRONT_MIRROR_HARD) == 1:
        rel = -rel
    return rel

def _is_front_180(rel_deg: float) -> bool:
    return abs(rel_deg) <= float(FRONT_HALF_DEG)

def _sector_name(rel_deg: float) -> Optional[str]:
    if not _is_front_180(rel_deg):
        return None
    c = float(SECTOR_CENTER_DEG)
    if rel_deg < -c:
        return "RIGHT"
    if rel_deg > c:
        return "LEFT"
    return "CENTER"


# =======================
# Reset logic
# =======================
def _reset_all_logic():
    """
    Reset về "như từ đầu" để robot tính lại fresh:
    - tắt avoid + reset streak + reset dir default
    - re-arm emergency gate
    - reset label debounce
    """
    global _avoid_active, _avoid_clear_streak, _avoid_dir, _avoid_start_ts, _avoid_last_switch_ts
    global _em_stop_armed, _em_stop_issued_this_event
    global _out_label, _pending_label, _pending_n

    with _avoid_lock:
        _avoid_active = False
        _avoid_clear_streak = 0
        _avoid_dir = "TURN_RIGHT"
        _avoid_start_ts = 0.0
        _avoid_last_switch_ts = 0.0

    with _em_lock:
        _em_stop_armed = True
        _em_stop_issued_this_event = False

    with _label_lock:
        _out_label = "STOP"
        _pending_label = ""
        _pending_n = 0


# =======================
# Avoid / Emergency gates
# =======================
def _avoid_set(active: bool, direction: Optional[str] = None):
    global _avoid_active, _avoid_clear_streak, _avoid_dir, _avoid_start_ts, _avoid_last_switch_ts
    with _avoid_lock:
        _avoid_active = bool(active)
        if direction in ("TURN_RIGHT", "TURN_LEFT"):
            _avoid_dir = direction
        if active:
            now = time.time()
            if _avoid_start_ts <= 0.0:
                _avoid_start_ts = now
            if _avoid_last_switch_ts <= 0.0:
                _avoid_last_switch_ts = now
        else:
            _avoid_clear_streak = 0
            _avoid_start_ts = 0.0
            _avoid_last_switch_ts = 0.0
            _avoid_dir = "TURN_RIGHT"

def _avoid_get() -> Tuple[bool, int, str, float, float]:
    with _avoid_lock:
        return bool(_avoid_active), int(_avoid_clear_streak), str(_avoid_dir), float(_avoid_start_ts), float(_avoid_last_switch_ts)

def _avoid_streak_inc():
    global _avoid_clear_streak
    with _avoid_lock:
        _avoid_clear_streak += 1

def _avoid_streak_reset():
    global _avoid_clear_streak
    with _avoid_lock:
        _avoid_clear_streak = 0

def _avoid_maybe_switch(new_dir: str):
    """
    Chỉ đổi hướng turn nếu:
    - đã qua commit time
    """
    global _avoid_dir, _avoid_last_switch_ts
    if new_dir not in ("TURN_LEFT", "TURN_RIGHT"):
        return
    now = time.time()
    with _avoid_lock:
        if not _avoid_active:
            return
        if now - _avoid_last_switch_ts < float(AVOID_COMMIT_SEC):
            return
        if new_dir != _avoid_dir:
            _avoid_dir = new_dir
            _avoid_last_switch_ts = now

def _em_is_armed() -> bool:
    with _em_lock:
        return bool(_em_stop_armed)

def _em_arm():
    global _em_stop_armed, _em_stop_issued_this_event
    with _em_lock:
        _em_stop_armed = True
        _em_stop_issued_this_event = False

def _em_disarm_after_issue():
    global _em_stop_armed, _em_stop_issued_this_event
    with _em_lock:
        _em_stop_armed = False
        _em_stop_issued_this_event = True

def _em_was_issued_this_event() -> bool:
    with _em_lock:
        return bool(_em_stop_issued_this_event)


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
            with lock:
                latest_points.append((theta, dist_m, q, ts))
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
# Front-180 snapshot + metrics
# =======================
def _collect_front_points(limit: int = 2400) -> Tuple[List[Dict[str, Any]], float]:
    with lock:
        pts = list(latest_points[-limit:])
        last_ts = float(latest_ts)

    now = time.time()
    out: List[Dict[str, Any]] = []
    for (theta, dist_m, q, ts) in pts:
        if (now - ts) > FRAME_SEC:
            continue
        if dist_m <= 0.02:
            continue

        rel = _rel_deg(theta)
        if not _is_front_180(rel):
            continue

        sec = _sector_name(rel)
        out.append({
            "theta": float(theta),
            "dist_m": float(dist_m),
            "dist_cm": float(dist_m * 100.0),
            "q": int(q),
            "ts": float(ts),
            "rel_deg": float(rel),
            "is_front_180": True,
            "k3_sector": sec,
        })

    return out, last_ts

def _min_front_any(front_points: List[Dict[str, Any]]) -> float:
    best = float("inf")
    for p in front_points:
        d = float(p["dist_m"])
        if d < best:
            best = d
    return best

def _sector_metrics(front_points: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets = {"LEFT": [], "CENTER": [], "RIGHT": []}
    for p in front_points:
        sec = p.get("k3_sector")
        if sec in buckets:
            buckets[sec].append(float(p["dist_m"]))

    out: Dict[str, Dict[str, Any]] = {}
    for name, ds in buckets.items():
        ds_sorted = sorted(ds)
        cnt = len(ds_sorted)
        min_d = ds_sorted[0] if cnt else float("inf")
        if cnt:
            k = min(K_NEAR, cnt)
            kmean = sum(ds_sorted[:k]) / float(k)
        else:
            kmean = float("inf")

        out[name] = {
            "count": int(cnt),
            "min_dist": float(min_d),
            "kmean_near": float(kmean),
        }
    return out

def _unknown(sec: Dict[str, Any]) -> bool:
    return int(sec.get("count", 0)) < int(MIN_SECTOR_POINTS)

def _sector_near(secs: Dict[str, Dict[str, Any]]) -> Tuple[bool, float]:
    """
    Rule mới theo bạn:
    Nếu trong 3 vùng có vật cản quá gần (<= OBSTACLE_NEAR_M) => STOP rồi TURN
    """
    mins = []
    for k in ("LEFT", "CENTER", "RIGHT"):
        d = float(secs.get(k, {}).get("min_dist", float("inf")))
        # nếu sector thiếu điểm thì coi như không tin (đỡ STOP do thiếu dữ liệu)
        if _unknown(secs.get(k, {})):
            continue
        mins.append(d)
    if not mins:
        return False, float("inf")
    m = min(mins)
    return (m <= float(OBSTACLE_NEAR_M)), m


# =======================
# Gap Planner + Corridor
# =======================
def _rotate_xy(x: float, y: float, heading_deg: float) -> Tuple[float, float]:
    # heading_deg: + là quay về bên trái
    r = math.radians(heading_deg)
    c = math.cos(r)
    s = math.sin(r)
    # (x',y') = R(heading)*(x,y)
    xr = x * c - y * s
    yr = x * s + y * c
    return xr, yr

def _corridor_min_x(front_points: List[Dict[str, Any]], heading_deg: float) -> float:
    """
    Hành lang hình chữ nhật:
    - chiều dài: LOOKAHEAD_M
    - nửa bề rộng: (ROBOT_WIDTH/2 + margin)
    Trả về min khoảng cách phía trước x (m) của điểm nào nằm trong corridor.
    Nếu không có điểm -> inf (coi như clear).
    """
    half_w = (float(ROBOT_WIDTH_M) * 0.5) + float(CORRIDOR_MARGIN_M)
    best = float("inf")

    for p in front_points:
        rel = float(p["rel_deg"])
        d = float(p["dist_m"])
        if not math.isfinite(d) or d <= 0:
            continue
        # đổi sang tọa độ robot frame (x forward, y left)
        rr = math.radians(rel)
        x = d * math.cos(rr)
        y = d * math.sin(rr)

        # đổi sang frame corridor (quay ngược heading để corridor nằm thẳng)
        xc, yc = _rotate_xy(x, y, -heading_deg)

        if xc <= 0.0 or xc > float(LOOKAHEAD_M):
            continue
        if abs(yc) <= half_w:
            if xc < best:
                best = xc

    return best

def _build_bins(front_points: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """
    bins cho -90..+90 (front180), bước BIN_DEG.
    dist_bin = min dist trong bin.
    center_deg_bin = list góc center của mỗi bin.
    """
    bin_deg = float(BIN_DEG)
    n = int(round((180.0 / bin_deg))) + 1  # -90..+90 inclusive
    centers = [-90.0 + i * bin_deg for i in range(n)]
    dists = [float("inf")] * n

    for p in front_points:
        rel = float(p["rel_deg"])
        d = float(p["dist_m"])
        if not math.isfinite(d) or d <= 0:
            continue
        if rel < -90.0 or rel > 90.0:
            continue
        idx = int(round((rel + 90.0) / bin_deg))
        if 0 <= idx < n:
            if d < dists[idx]:
                dists[idx] = d

    return dists, centers

def _find_gaps(dists: List[float], centers: List[float]) -> List[Dict[str, Any]]:
    """
    Gap = đoạn liên tục các bin "free".

    Free condition: dist >= OBSTACLE_NEAR_M (đủ xa để đi qua khe).
    Nhưng "khe hẹp" sẽ được lọc theo chiều ngang robot:
      - ước lượng width_m tại dist_ref (điểm thắt gần nhất trong gap, nhưng không vượt LOOKAHEAD_M)
      - yêu cầu width_m >= (robot_width + 2*margin + extra)
    """
    free_th = float(OBSTACLE_NEAR_M)
    bin_deg = float(BIN_DEG)
    req_w = _required_clear_width_m()

    gaps: List[Dict[str, Any]] = []
    i = 0
    n = len(dists)
    while i < n:
        if (not math.isfinite(dists[i])) or (dists[i] < free_th):
            i += 1
            continue
        j = i
        while j < n and (math.isfinite(dists[j]) and dists[j] >= free_th):
            j += 1

        start_deg = centers[i] - 0.5 * bin_deg
        end_deg = centers[j - 1] + 0.5 * bin_deg
        width_deg = end_deg - start_deg

        # clearance đại diện = min(dist) trong gap (điểm thắt gần nhất)
        clearance = min(dists[i:j]) if i < j else float("inf")

        # dist_ref = điểm thắt gần nhất, nhưng không vượt lookahead
        dist_ref = float(min(float(LOOKAHEAD_M), float(clearance if math.isfinite(clearance) else float(LOOKAHEAD_M))))

        # min deg để đạt req_w tại dist_ref
        need_deg_dyn = _min_gap_width_deg(dist_ref)
        need_deg = float(max(float(GAP_MIN_WIDTH_DEG), need_deg_dyn))

        # width_m tại dist_ref
        width_m = _gap_width_m(width_deg, dist_ref)

        # pass if đủ độ và đủ mét
        if (width_deg >= need_deg) and (width_m >= req_w):
            best_deg = (start_deg + end_deg) * 0.5
            gaps.append({
                "start_deg": float(start_deg),
                "end_deg": float(end_deg),
                "width_deg": float(width_deg),
                "best_deg": float(best_deg),
                "clearance_m": float(clearance),
                "dist_ref_m": float(dist_ref),
                "gap_width_m": float(width_m),
                "required_width_m": float(req_w),
                "need_deg": float(need_deg),
                "need_deg_dyn": float(need_deg_dyn),
                "i0": int(i),
                "i1": int(j - 1),
            })

        i = j

    return gaps

def _score_gap(g: Dict[str, Any]) -> float:
    """
    Score heuristic:
    - khe rộng theo mét (ở dist_ref) tốt hơn
    - clearance lớn tốt hơn
    - gần hướng thẳng (0 độ) tốt hơn (đỡ lắc)
    """
    width_m = float(g.get("gap_width_m", 0.0))
    req_w = float(g.get("required_width_m", _required_clear_width_m()))
    clear = float(g.get("clearance_m", 0.0))
    best = float(g.get("best_deg", 0.0))

    # normalize width: >= 2*req_w coi như max
    w_n = max(0.0, min(1.0, width_m / max(1e-6, (2.0 * req_w))))
    c_n = max(0.0, min(1.0, clear / max(1.0, CLEAR_GO_M)))
    s_n = 1.0 - max(0.0, min(1.0, abs(best) / 90.0))

    return 0.50 * w_n + 0.30 * c_n + 0.20 * s_n

def _pick_best_heading(front_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Trả:
    - best_heading_deg, best_score
    - top_candidates (list 3)
    - bins preview
    """
    dists, centers = _build_bins(front_points)
    gaps = _find_gaps(dists, centers)
    for g in gaps:
        g["score"] = float(_score_gap(g))

    gaps_sorted = sorted(gaps, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    top = gaps_sorted[:3]

    if top:
        best = top[0]
        return {
            "best_heading_deg": float(best["best_deg"]),
            "best_score": float(best["score"]),
            "required_width_m": float(_required_clear_width_m()),
            "candidates": [{
                "best_deg": float(x["best_deg"]),
                "score": float(x["score"]),
                "width_deg": float(x["width_deg"]),
                "gap_width_m": float(x.get("gap_width_m", 0.0)),
                "required_width_m": float(x.get("required_width_m", _required_clear_width_m())),
                "need_deg": float(x.get("need_deg", 0.0)),
                "dist_ref_m": float(x.get("dist_ref_m", 0.0)),
                "clearance_m": float(x["clearance_m"]),
                "start_deg": float(x["start_deg"]),
                "end_deg": float(x["end_deg"]),
            } for x in top],
            "bins": {
                "bin_deg": float(BIN_DEG),
                "centers": centers,
                "dists": dists,
            }
        }

    # fallback: nếu không có gap đủ rộng, chọn hướng nào corridor tốt hơn (±45)
    c0 = _corridor_min_x(front_points, 0.0)
    cl = _corridor_min_x(front_points, +45.0)
    cr = _corridor_min_x(front_points, -45.0)

    best_h = 0.0
    best_s = 0.0
    if cl >= cr and cl > c0:
        best_h, best_s = +45.0, 0.1
    elif cr > cl and cr > c0:
        best_h, best_s = -45.0, 0.1

    return {
        "best_heading_deg": float(best_h),
        "best_score": float(best_s),
        "required_width_m": float(_required_clear_width_m()),
        "candidates": [],
        "bins": {
            "bin_deg": float(BIN_DEG),
            "centers": centers,
            "dists": dists,
        }
    }

def _heading_to_label(heading_deg: float) -> str:
    if abs(float(heading_deg)) <= float(TURN_DEADBAND_DEG):
        return "GO_STRAIGHT"
    return "TURN_LEFT" if float(heading_deg) > 0 else "TURN_RIGHT"


# =======================
# Label debounce
# =======================
def _smooth_label(raw: str) -> str:
    global _out_label, _pending_label, _pending_n
    raw = str(raw or "STOP")

    with _label_lock:
        if raw == _out_label:
            _pending_label = ""
            _pending_n = 0
            return _out_label

        if raw == _pending_label:
            _pending_n += 1
        else:
            _pending_label = raw
            _pending_n = 1

        if _pending_n >= int(LABEL_CONFIRM_N):
            _out_label = raw
            _pending_label = ""
            _pending_n = 0

        return _out_label


# =======================
# Decision logic (Gap Planner)
# - NEAR in any sector => STOP then TURN
# - Avoid: turn theo hướng tốt, có commit, clear => RESET STATE => GO_STRAIGHT
# - Normal: ưu tiên corridor thẳng, không clear thì steer theo gap heading
# =======================
def _pack_decision(label: str, reason: str, secs: Dict[str, Dict[str, Any]], min_front: float,
                   avoid_active: bool, clear_streak: int, avoid_dir: str, avoid_age: float,
                   planner_dbg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "label": label,
        "reason": reason,
        "ts": time.time(),
        "debug": {
            "front": {
                "center": FRONT_CENTER_DEG_HARD,
                "mirror": FRONT_MIRROR_HARD,
                "flip": FRONT_FLIP_HARD,
                "frame_sec": float(FRAME_SEC),
            },
            "sectors": {
                "CENTER": secs["CENTER"],
                "LEFT": secs["LEFT"],
                "RIGHT": secs["RIGHT"],
            },
            "min_front": float(min_front),
            "avoid": {
                "active": bool(avoid_active),
                "clear_streak": int(clear_streak),
                "dir": avoid_dir,
                "age_s": float(avoid_age),
                "max_s": float(AVOID_MAX_SEC),
                "commit_s": float(AVOID_COMMIT_SEC),
                "switch_hyst": float(AVOID_SWITCH_HYST),
            },
            "emergency": {"armed": _em_is_armed(), "issued_this_event": _em_was_issued_this_event()},
            "thresholds": {
                "clear_go_m": float(CLEAR_GO_M),
                "obstacle_near_m": float(OBSTACLE_NEAR_M),
                "clear_streak_n": int(CLEAR_STREAK_N),
                "min_sector_points": int(MIN_SECTOR_POINTS),
                "k_near": int(K_NEAR),
                "label_confirm_n": int(LABEL_CONFIRM_N),
                "emergency_stop_m": float(EMERGENCY_STOP_M),
                "emergency_rearm_m": float(EMERGENCY_REARM_M),
                "robot_width_m": float(ROBOT_WIDTH_M),
                "corridor_margin_m": float(CORRIDOR_MARGIN_M),
                "gap_extra_m": float(GAP_EXTRA_M),
                "required_clear_width_m": float(_required_clear_width_m()),
                "lookahead_m": float(LOOKAHEAD_M),
                "bin_deg": float(BIN_DEG),
                "gap_min_width_deg_floor": float(GAP_MIN_WIDTH_DEG),
            },
            "planner": planner_dbg or {},
        }
    }

def _compute_decision() -> Dict[str, Any]:
    now = time.time()
    front_pts, last_ts = _collect_front_points(limit=2400)

    if (not front_pts) or ((now - last_ts) > 2.0):
        _reset_all_logic()
        raw = {"ok": False, "label": "STOP", "reason": "no_recent_front180_points", "ts": now}
        raw["label"] = _smooth_label(raw["label"])
        return raw

    secs = _sector_metrics(front_pts)
    min_front = _min_front_any(front_pts)

    # planner snapshot
    plan = _pick_best_heading(front_pts)
    best_heading = float(plan.get("best_heading_deg", 0.0))
    best_label = _heading_to_label(best_heading)
    best_score = float(plan.get("best_score", 0.0))

    # corridors debug (planned + others)
    cand_headings = [best_heading]
    for c in (plan.get("candidates") or []):
        hd = float(c.get("best_deg", 0.0))
        if hd not in cand_headings:
            cand_headings.append(hd)
        if len(cand_headings) >= 3:
            break
    if 0.0 not in cand_headings:
        cand_headings.append(0.0)

    corridors_dbg = []
    for hd in cand_headings[:4]:
        mx = _corridor_min_x(front_pts, hd)
        corridors_dbg.append({
            "heading_deg": float(hd),
            "min_x_m": float(mx if math.isfinite(mx) else 999.0),
            "is_clear": bool((not math.isfinite(mx)) or (mx >= float(CLEAR_GO_M))),
        })

    near_any, near_min = _sector_near(secs)

    # ========== Emergency STOP one-shot ==========
    if (min_front <= float(EMERGENCY_STOP_M)) and _em_is_armed():
        _em_disarm_after_issue()
        turn_dir = "TURN_LEFT" if best_heading > 0 else "TURN_RIGHT"
        _avoid_set(True, turn_dir)

        avoid_active, streak, avoid_dir, start_ts, _swts = _avoid_get()
        avoid_age = (time.time() - start_ts) if start_ts > 0 else 0.0

        planner_dbg = {
            "best_heading_deg": best_heading,
            "best_label": best_label,
            "best_score": best_score,
            "near_any": near_any,
            "near_min_m": float(near_min),
            "corridors": corridors_dbg,
            "candidates": plan.get("candidates", []),
            "required_clear_width_m": float(_required_clear_width_m()),
        }

        raw = _pack_decision(
            label="STOP",
            reason=f"emergency_stop_once(min_front={min_front:.3f}) -> enter_avoid({turn_dir})",
            secs=secs,
            min_front=min_front,
            avoid_active=True,
            clear_streak=0,
            avoid_dir=turn_dir,
            avoid_age=avoid_age,
            planner_dbg=planner_dbg,
        )
        raw["label"] = _smooth_label(raw["label"])
        return raw

    # re-arm emergency
    if (min_front > float(EMERGENCY_REARM_M)) and (not _em_is_armed()):
        _em_arm()

    # ========== Rule: NEAR ở bất kỳ vùng nào => STOP rồi TURN ==========
    avoid_active, streak, avoid_dir, start_ts, last_sw = _avoid_get()
    avoid_age = (time.time() - start_ts) if start_ts > 0 else 0.0

    if (not avoid_active) and near_any:
        turn_dir = "TURN_LEFT" if best_heading > 0 else "TURN_RIGHT"
        _avoid_set(True, turn_dir)

        planner_dbg = {
            "best_heading_deg": best_heading,
            "best_label": best_label,
            "best_score": best_score,
            "near_any": near_any,
            "near_min_m": float(near_min),
            "corridors": corridors_dbg,
            "candidates": plan.get("candidates", []),
            "required_clear_width_m": float(_required_clear_width_m()),
        }

        raw = _pack_decision(
            label="STOP",
            reason=f"near_obstacle(min={near_min:.3f} <= {OBSTACLE_NEAR_M:.2f}) -> STOP then enter_avoid({turn_dir})",
            secs=secs,
            min_front=min_front,
            avoid_active=True,
            clear_streak=0,
            avoid_dir=turn_dir,
            avoid_age=0.0,
            planner_dbg=planner_dbg,
        )
        raw["label"] = _smooth_label(raw["label"])
        return raw

    # ========== Avoid mode ==========
    avoid_active, streak, avoid_dir, start_ts, last_sw = _avoid_get()
    avoid_age = (time.time() - start_ts) if start_ts > 0 else 0.0

    if avoid_active and avoid_age >= float(AVOID_MAX_SEC):
        _reset_all_logic()
        avoid_active, streak, avoid_dir, start_ts, last_sw = _avoid_get()
        avoid_age = 0.0

    cmin0 = _corridor_min_x(front_pts, 0.0)
    corridor_clear0 = (not math.isfinite(cmin0)) or (cmin0 >= float(CLEAR_GO_M))
    clear_now = corridor_clear0 and (float(min_front) >= float(CLEAR_GO_M))

    if avoid_active:
        if clear_now:
            _avoid_streak_inc()
        else:
            _avoid_streak_reset()

        avoid_active2, streak2, avoid_dir2, start_ts2, last_sw2 = _avoid_get()
        avoid_age2 = (time.time() - start_ts2) if start_ts2 > 0 else 0.0

        planner_dbg = {
            "best_heading_deg": best_heading,
            "best_label": best_label,
            "best_score": best_score,
            "near_any": near_any,
            "near_min_m": float(near_min),
            "corridor_min_x0_m": float(cmin0 if math.isfinite(cmin0) else 999.0),
            "corridor_clear0": bool(corridor_clear0),
            "clear_now": bool(clear_now),
            "corridors": corridors_dbg,
            "candidates": plan.get("candidates", []),
            "required_clear_width_m": float(_required_clear_width_m()),
        }

        if streak2 >= int(CLEAR_STREAK_N):
            _reset_all_logic()
            raw = _pack_decision(
                label="GO_STRAIGHT",
                reason="avoid_clear_confirmed -> RESET_STATE -> go_straight",
                secs=secs,
                min_front=min_front,
                avoid_active=False,
                clear_streak=streak2,
                avoid_dir=avoid_dir2,
                avoid_age=avoid_age2,
                planner_dbg=planner_dbg,
            )
            raw["label"] = _smooth_label(raw["label"])
            return raw

        desired_turn = "TURN_LEFT" if best_heading > 0 else "TURN_RIGHT"
        if desired_turn != avoid_dir2 and best_score >= float(AVOID_SWITCH_HYST):
            _avoid_maybe_switch(desired_turn)
            avoid_active2, streak2, avoid_dir2, start_ts2, last_sw2 = _avoid_get()

        raw = _pack_decision(
            label=avoid_dir2,
            reason="avoid_mode_turn_until_clear (NEAR->STOP triggered)",
            secs=secs,
            min_front=min_front,
            avoid_active=True,
            clear_streak=streak2,
            avoid_dir=avoid_dir2,
            avoid_age=avoid_age2,
            planner_dbg=planner_dbg,
        )
        raw["label"] = _smooth_label(raw["label"])
        return raw

    # ========== Normal mode ==========
    planner_dbg = {
        "best_heading_deg": best_heading,
        "best_label": best_label,
        "best_score": best_score,
        "near_any": near_any,
        "near_min_m": float(near_min),
        "corridor_min_x0_m": float(cmin0 if math.isfinite(cmin0) else 999.0),
        "corridor_clear0": bool(corridor_clear0),
        "corridors": corridors_dbg,
        "candidates": plan.get("candidates", []),
        "required_clear_width_m": float(_required_clear_width_m()),
    }

    if corridor_clear0 and (float(min_front) >= float(CLEAR_GO_M)):
        raw = _pack_decision(
            label="GO_STRAIGHT",
            reason="corridor_forward_clear -> go_straight",
            secs=secs,
            min_front=min_front,
            avoid_active=False,
            clear_streak=0,
            avoid_dir="TURN_RIGHT",
            avoid_age=0.0,
            planner_dbg=planner_dbg,
        )
        raw["label"] = _smooth_label(raw["label"])
        return raw

    steer = best_label
    if steer == "GO_STRAIGHT":
        cl = _corridor_min_x(front_pts, +35.0)
        cr = _corridor_min_x(front_pts, -35.0)
        steer = "TURN_LEFT" if cl >= cr else "TURN_RIGHT"
        planner_dbg["fallback_turn"] = {"cl_m": float(cl if math.isfinite(cl) else 999.0), "cr_m": float(cr if math.isfinite(cr) else 999.0)}

    raw = _pack_decision(
        label=steer,
        reason=f"not_clear_forward -> steer_by_gap_planner({steer})",
        secs=secs,
        min_front=min_front,
        avoid_active=False,
        clear_streak=0,
        avoid_dir="TURN_RIGHT",
        avoid_age=0.0,
        planner_dbg=planner_dbg,
    )
    raw["label"] = _smooth_label(raw["label"])
    return raw


# =======================
# Rendering (map + corridors)
# =======================
def _render_map_png(decision: Dict[str, Any], front_points: List[Dict[str, Any]]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    import io

    W = H = int(VIEW_SIZE_PX)
    img = Image.new("RGB", (W, H), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    cx, cy = W // 2, H // 2
    ppm = (W * 0.45) / max(0.5, float(VIEW_RANGE_M))

    rr = int(VIEW_RANGE_M * ppm)
    bbox = (cx - rr, cy - rr, cx + rr, cy + rr)
    draw.arc(bbox, start=0, end=180, fill=(160, 160, 160), width=3)

    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(20, 20, 20))

    def xy_to_px(x_fwd: float, y_left: float) -> Tuple[int, int]:
        px = int(cx + y_left * ppm)
        py = int(cy - x_fwd * ppm)
        return px, py

    def rel_to_px(rel_deg: float, dist_m: float) -> Tuple[int, int]:
        r = math.radians(rel_deg)
        x_fwd = dist_m * math.cos(r)
        y_left = dist_m * math.sin(r)
        return xy_to_px(x_fwd, y_left)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    ring_px = int(SECTOR_RING_M * ppm)
    bbox2 = (cx - ring_px, cy - ring_px, cx + ring_px, cy + ring_px)

    label = str(decision.get("label", "STOP"))
    selected_sector = None
    if label == "GO_STRAIGHT":
        selected_sector = "CENTER"
    elif label == "TURN_RIGHT":
        selected_sector = "RIGHT"
    elif label == "TURN_LEFT":
        selected_sector = "LEFT"

    sector_defs = {
        "RIGHT": (-90.0, -SECTOR_CENTER_DEG),
        "CENTER": (-SECTOR_CENTER_DEG, +SECTOR_CENTER_DEG),
        "LEFT": (+SECTOR_CENTER_DEG, +90.0),
    }

    def rel_to_pil(rel_deg: float) -> float:
        return float(90.0 + rel_deg)

    for name, (a0, a1) in sector_defs.items():
        start = rel_to_pil(a0)
        end = rel_to_pil(a1)
        fill = (0, 255, 0, SECTOR_ALPHA) if selected_sector == name else (255, 0, 0, SECTOR_ALPHA)
        od.pieslice(bbox2, start=start, end=end, fill=fill)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    for p in front_points:
        d = float(p["dist_m"])
        rel = float(p["rel_deg"])
        if d > VIEW_RANGE_M:
            continue
        px, py = rel_to_px(rel, d)
        if 0 <= px < W and 0 <= py < H:
            img.putpixel((px, py), (80, 80, 80))

    dbg = decision.get("debug", {}) if isinstance(decision, dict) else {}
    planner = dbg.get("planner", {}) if isinstance(dbg, dict) else {}
    corridors = planner.get("corridors", []) if isinstance(planner, dict) else []

    half_w = (float(ROBOT_WIDTH_M) * 0.5) + float(CORRIDOR_MARGIN_M)
    look = float(LOOKAHEAD_M)

    def corridor_poly(heading_deg: float) -> List[Tuple[int, int]]:
        corners = [
            (0.0, -half_w),
            (look, -half_w),
            (look, +half_w),
            (0.0, +half_w),
        ]
        pts = []
        for (xh, yh) in corners:
            x, y = _rotate_xy(xh, yh, heading_deg)
            pts.append(xy_to_px(x, y))
        return pts

    cor_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cor_ov)

    for i, c in enumerate(corridors):
        hd = float(c.get("heading_deg", 0.0))
        poly = corridor_poly(hd)

        if i == 0:
            fill = (0, 200, 0, 70)      # xanh (corridor chuẩn bị đi)
            outline = (0, 120, 0, 140)
        else:
            fill = (255, 235, 59, 45)   # vàng nhạt (corridor khác)
            outline = (180, 160, 0, 110)

        cd.polygon(poly, fill=fill, outline=outline)

    img = Image.alpha_composite(img.convert("RGBA"), cor_ov).convert("RGB")
    draw = ImageDraw.Draw(img)

    if label == "GO_STRAIGHT":
        ax, ay = rel_to_px(0.0, min(1.0, VIEW_RANGE_M * 0.9))
        draw.line((cx, cy, ax, ay), fill=(0, 0, 0), width=4)
    elif label == "TURN_RIGHT":
        ax, ay = rel_to_px(-45.0, min(1.0, VIEW_RANGE_M * 0.9))
        draw.line((cx, cy, ax, ay), fill=(0, 0, 0), width=4)
    elif label == "TURN_LEFT":
        ax, ay = rel_to_px(+45.0, min(1.0, VIEW_RANGE_M * 0.9))
        draw.line((cx, cy, ax, ay), fill=(0, 0, 0), width=4)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = None
        font_small = None

    secdbg = dbg.get("sectors", {}) if isinstance(dbg, dict) else {}

    def fmt_dist(d: float) -> str:
        if not math.isfinite(d):
            return "inf"
        return f"{d*100:.0f}cm"

    dL = float(secdbg.get("LEFT", {}).get("min_dist", float("inf")))
    dC = float(secdbg.get("CENTER", {}).get("min_dist", float("inf")))
    dR = float(secdbg.get("RIGHT", {}).get("min_dist", float("inf")))

    for name, rel, d in [("LEFT", +60.0, dL), ("CENTER", 0.0, dC), ("RIGHT", -60.0, dR)]:
        tx, ty = rel_to_px(rel, min(SECTOR_RING_M * 0.85, VIEW_RANGE_M))
        draw.text((tx - 35, ty - 10), f"{name}:{fmt_dist(d)}", fill=(0, 0, 0), font=font_small)

    avoid_dbg = dbg.get("avoid", {}) if isinstance(dbg, dict) else {}
    cor0 = planner.get("corridor_min_x0_m", None) if isinstance(planner, dict) else None

    req_w = _required_clear_width_m()

    hud = [
        f"label: {label}",
        f"front_center={FRONT_CENTER_DEG_HARD:.1f} mirror={FRONT_MIRROR_HARD} flip={FRONT_FLIP_HARD}",
        f"frame_sec={FRAME_SEC:.2f} bin={BIN_DEG:.1f}deg lookahead={LOOKAHEAD_M:.2f}m",
        f"clear_go={CLEAR_GO_M:.2f}m obstacle_near={OBSTACLE_NEAR_M:.2f}m em_stop={EMERGENCY_STOP_M:.2f}m",
        f"robot_w={ROBOT_WIDTH_M*100:.0f}cm margin={CORRIDOR_MARGIN_M*100:.0f}cm extra={GAP_EXTRA_M*100:.0f}cm req_gap={req_w*100:.0f}cm",
        f"avoid: active={avoid_dbg.get('active')} dir={avoid_dbg.get('dir')} streak={avoid_dbg.get('clear_streak')} age={avoid_dbg.get('age_s')}",
        f"L={fmt_dist(dL)} C={fmt_dist(dC)} R={fmt_dist(dR)} min_front={fmt_dist(float(dbg.get('min_front', float('inf'))))}",
        f"best_heading={planner.get('best_heading_deg', 0.0)} deg score={planner.get('best_score', 0.0)}",
        f"corridor0_min_x={(cor0 if cor0 is not None else '-')}",
        f"reason: {str(decision.get('reason',''))[:90]}",
    ]
    y0 = 10
    for s in hud:
        draw.text((10, y0), s, fill=(0, 0, 0), font=font)
        y0 += 22

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =======================
# Payload builders
# =======================
def _build_points_payload(limit: int = 1600) -> Dict[str, Any]:
    front_pts, last_ts = _collect_front_points(limit=limit)
    return {
        "ok": True,
        "ts": time.time(),
        "last_point_ts": last_ts,
        "n": len(front_pts),
        "points": front_pts,
        "frame": {
            "front_center_deg_used": float(FRONT_CENTER_DEG_HARD),
            "front_half_deg": float(FRONT_HALF_DEG),
            "mirror": int(FRONT_MIRROR_HARD),
            "flip": int(FRONT_FLIP_HARD),
            "frame_sec": float(FRAME_SEC),
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

def map_worker():
    while True:
        with decision_state_lock:
            dec = dict(latest_decision_full)
        front_pts, _ = _collect_front_points(limit=2400)
        try:
            png = _render_map_png(decision=dec, front_points=front_pts)
            with cache_lock:
                _map_cache["ts"] = time.time()
                _map_cache["png"] = png
        except Exception:
            pass
        time.sleep(THROTTLE_S)


# =======================
# ROUTES (giữ y chang format/endpoint cũ)
# =======================
@app.get("/api/status")
def api_status():
    with lock:
        pts_n = len(latest_points)
        ts = float(latest_ts)

    with decision_state_lock:
        dec = dict(latest_decision_full)
        lbl = latest_decision_label

    avoid_active, streak, avoid_dir, start_ts, last_sw = _avoid_get()
    avoid_age = (time.time() - start_ts) if start_ts > 0 else 0.0
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

        "front": {
            "center": FRONT_CENTER_DEG_HARD,
            "mirror": FRONT_MIRROR_HARD,
            "flip": FRONT_FLIP_HARD,
            "frame_sec": float(FRAME_SEC),
        },

        "avoid": {"active": bool(avoid_active), "clear_streak": int(streak), "dir": avoid_dir, "age_s": float(avoid_age)},
        "emergency": {"armed": _em_is_armed(), "issued_this_event": _em_was_issued_this_event()},

        "thresholds": {
            "clear_go_m": float(CLEAR_GO_M),
            "obstacle_near_m": float(OBSTACLE_NEAR_M),
            "clear_streak_n": int(CLEAR_STREAK_N),
            "min_sector_points": int(MIN_SECTOR_POINTS),
            "k_near": int(K_NEAR),
            "label_confirm_n": int(LABEL_CONFIRM_N),
            "avoid_max_sec": float(AVOID_MAX_SEC),
            "emergency_stop_m": float(EMERGENCY_STOP_M),
            "emergency_rearm_m": float(EMERGENCY_REARM_M),
            "robot_width_m": float(ROBOT_WIDTH_M),
            "corridor_margin_m": float(CORRIDOR_MARGIN_M),
            "gap_extra_m": float(GAP_EXTRA_M),
            "required_clear_width_m": float(_required_clear_width_m()),
            "lookahead_m": float(LOOKAHEAD_M),
            "bin_deg": float(BIN_DEG),
            "gap_min_width_deg_floor": float(GAP_MIN_WIDTH_DEG),
        },

        "latest_label": lbl,
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

@app.get("/api/decision")
def api_decision():
    with decision_state_lock:
        payload = dict(latest_decision_full)
    return jsonify(payload)

@app.get("/api/decision_label")
def api_decision_label():
    with decision_state_lock:
        lbl = latest_decision_label
    return Response(str(lbl), mimetype="text/plain")

@app.get("/api/map.png")
def api_map_png():
    with cache_lock:
        png = _map_cache["png"]
    if png is None:
        with decision_state_lock:
            dec = dict(latest_decision_full)
        front_pts, _ = _collect_front_points(limit=2400)
        png = _render_map_png(decision=dec, front_points=front_pts)
        with cache_lock:
            _map_cache["png"] = png
            _map_cache["ts"] = time.time()
    return Response(png, mimetype="image/png")

@app.post("/api/reset_logic")
def api_reset_logic():
    _reset_all_logic()
    return jsonify({"ok": True})

@app.get("/dashboard")
def dashboard():
    html = f"""
    <html>
      <head>
        <title>LiDAR Front-180 Map (Gap Planner)</title>
        <style>
          body {{ font-family: Arial; margin: 12px; }}
          .row {{ display:flex; gap:12px; align-items:flex-start; }}
          img {{ border:1px solid #ccc; border-radius:8px; }}
          .box {{ padding:10px; border:1px solid #ddd; border-radius:8px; min-width: 420px; }}
          .mono {{ font-family: monospace; white-space: pre; }}
          button {{ padding:6px 10px; }}
        </style>
      </head>
      <body>
        <h3>LiDAR Front-180 Map (Gap Planner) — NEAR => STOP then TURN, clear => RESET STATE</h3>
        <div class="row">
          <img id="map" src="/api/map.png?ts={time.time()}" width="{VIEW_SIZE_PX}" height="{VIEW_SIZE_PX}"/>
          <div class="box">
            <div><b>Orientation (locked)</b></div>
            <div class="mono">{{
  "center": {FRONT_CENTER_DEG_HARD},
  "mirror": {FRONT_MIRROR_HARD},
  "flip": {FRONT_FLIP_HARD}
}}</div>
            <button onclick="resetLogic()">Reset Logic</button>
            <hr/>
            <div><b>Decision</b></div>
            <div id="label" class="mono">loading...</div>
            <hr/>
            <div><b>Status</b></div>
            <div id="status" class="mono">loading...</div>
          </div>
        </div>

        <script>
          async function resetLogic() {{
            try {{
              await fetch('/api/reset_logic', {{ method:'POST' }});
            }} catch(e) {{}}
          }}

          async function tick() {{
            try {{
              const d = await fetch('/api/decision').then(r=>r.json());
              const s = await fetch('/api/status').then(r=>r.json());

              document.getElementById('label').textContent =
                JSON.stringify({{
                  label: d.label,
                  reason: d.reason,
                  sectors: (d.debug||{{}}).sectors,
                  min_front: (d.debug||{{}}).min_front,
                  avoid: (d.debug||{{}}).avoid,
                  emergency: (d.debug||{{}}).emergency,
                  thresholds: (d.debug||{{}}).thresholds,
                  planner: (d.debug||{{}}).planner
                }}, null, 2);

              document.getElementById('status').textContent =
                JSON.stringify({{
                  running: s.running,
                  age_s: s.age_s,
                  front: s.front,
                  avoid: s.avoid,
                  thresholds: s.thresholds
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

@app.get("/")
def home():
    return Response(
        "<h3>lidarhub_front180_gap_planner running</h3>"
        "<ul>"
        "<li><a href='/dashboard'>/dashboard</a></li>"
        "<li>/api/map.png</li>"
        "<li>/take_lidar_data</li>"
        "<li>/api/decision</li>"
        "<li>/api/decision_label</li>"
        "<li>/api/status</li>"
        "<li>/api/reset_logic (POST)</li>"
        "</ul>",
        mimetype="text/html"
    )


# =======================
# Main
# =======================
def main():
    _reset_all_logic()
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    threading.Thread(target=points_worker, daemon=True).start()
    threading.Thread(target=decision_worker, daemon=True).start()
    threading.Thread(target=map_worker, daemon=True).start()
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
