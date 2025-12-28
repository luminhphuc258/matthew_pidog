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
POINT_LIMIT = int(os.environ.get("POINT_LIMIT", "4000"))

THROTTLE_MS = int(os.environ.get("THROTTLE_MS", "120"))
THROTTLE_S = THROTTLE_MS / 1000.0

# ===== FRONT definition =====
FRONT_CENTER_DEG = float(os.environ.get("FRONT_CENTER_DEG", "0.0"))

# 1 nếu bị đảo trái/phải
FRONT_MIRROR = int(os.environ.get("FRONT_MIRROR", "0"))

# 1 nếu đang bị đảo trước/sau (front/back) 180deg  <-- FIX lỗi bạn đang gặp
FRONT_FLIP = int(os.environ.get("FRONT_FLIP", "0"))

# khóa không cho dashboard đổi front_center (nếu muốn cố định)
LOCK_FRONT_CENTER = int(os.environ.get("LOCK_FRONT_CENTER", "0"))

# front 180 = ±90
FRONT_HALF_DEG = float(os.environ.get("FRONT_HALF_DEG", "90.0"))

# ===== Frame window (tránh smear khi robot quay) =====
FRAME_SEC = float(os.environ.get("FRAME_SEC", "0.18"))  # dùng cho decision/map
RECENT_SEC = float(os.environ.get("RECENT_SEC", "0.60"))  # buffer chung (không còn dùng trực tiếp trong decision)

# ===== k=3 sectors trong front 180 =====
SECTOR_CENTER_DEG = float(os.environ.get("SECTOR_CENTER_DEG", "30.0"))  # CENTER = [-30..+30]
MIN_SECTOR_POINTS = int(os.environ.get("MIN_SECTOR_POINTS", "30"))      # ít điểm => coi như không an toàn

# ===== Behavior thresholds =====
OBSTACLE_NEAR_M = float(os.environ.get("OBSTACLE_NEAR_M", "0.70"))
CLEAR_GO_M = float(os.environ.get("CLEAR_GO_M", "0.95"))
CLEAR_STREAK_N = int(os.environ.get("CLEAR_STREAK_N", "3"))

EMERGENCY_STOP_M = float(os.environ.get("EMERGENCY_STOP_M", "0.25"))
EMERGENCY_REARM_M = float(os.environ.get("EMERGENCY_REARM_M", "0.45"))

# ===== Rendering =====
VIEW_SIZE_PX = int(os.environ.get("VIEW_SIZE_PX", "720"))
VIEW_RANGE_M = float(os.environ.get("VIEW_RANGE_M", "3.5"))
SECTOR_RING_M = float(os.environ.get("SECTOR_RING_M", "1.25"))
SECTOR_ALPHA = int(os.environ.get("SECTOR_ALPHA", "90"))

# Read ultra_simple lines
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
latest_decision_label: str = "STOP"
latest_decision_full: Dict[str, Any] = {"ok": False, "label": "STOP", "reason": "init", "ts": 0.0}

_front_lock = threading.Lock()
_front_center_deg: float = FRONT_CENTER_DEG
_front_mirror: int = FRONT_MIRROR
_front_flip: int = FRONT_FLIP

# Avoid state: TURN_RIGHT until clear
_avoid_lock = threading.Lock()
_avoid_active: bool = False
_avoid_clear_streak: int = 0

# Emergency STOP one-shot gate
_em_lock = threading.Lock()
_em_stop_armed: bool = True
_em_stop_issued_this_event: bool = False


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

def _get_front_center_deg() -> float:
    with _front_lock:
        return float(_front_center_deg)

def _get_front_mirror() -> int:
    with _front_lock:
        return int(_front_mirror)

def _get_front_flip() -> int:
    with _front_lock:
        return int(_front_flip)

def _set_front_center_deg(v: float):
    global _front_center_deg
    if LOCK_FRONT_CENTER:
        return
    with _front_lock:
        _front_center_deg = float(_wrap_deg(v))

def _set_front_mirror(v: int):
    global _front_mirror
    with _front_lock:
        _front_mirror = 1 if int(v) else 0

def _set_front_flip(v: int):
    global _front_flip
    with _front_lock:
        _front_flip = 1 if int(v) else 0

def _rel_deg(theta_deg: float, front_center_abs: float) -> float:
    """
    rel in [-180..180]
    + là LEFT, - là RIGHT
    """
    fc = _wrap_deg(front_center_abs + (180.0 if _get_front_flip() == 1 else 0.0))
    rel = _wrap_rel_deg(theta_deg - fc)

    if _get_front_mirror() == 1:
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
# Avoid / Emergency gates
# =======================
def _avoid_set(active: bool):
    global _avoid_active, _avoid_clear_streak
    with _avoid_lock:
        _avoid_active = bool(active)
        if not active:
            _avoid_clear_streak = 0

def _avoid_get() -> Tuple[bool, int]:
    with _avoid_lock:
        return bool(_avoid_active), int(_avoid_clear_streak)

def _avoid_clear_streak_inc():
    global _avoid_clear_streak
    with _avoid_lock:
        _avoid_clear_streak += 1

def _avoid_clear_streak_reset():
    global _avoid_clear_streak
    with _avoid_lock:
        _avoid_clear_streak = 0

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
# Front-180 filter + sector stats
# =======================
def _collect_front_points(limit: int = 2400) -> Tuple[List[Dict[str, Any]], float]:
    """
    Return list of dict points (front 180 only) + last_ts
    Each point includes: theta, dist_m, q, ts, rel_deg, sector
    Uses FRAME_SEC (snapshot window) to avoid smear while turning.
    """
    with lock:
        pts = list(latest_points[-limit:])
        last_ts = float(latest_ts)

    front_abs = _get_front_center_deg()
    now = time.time()

    out: List[Dict[str, Any]] = []
    for (theta, dist_m, q, ts) in pts:
        if (now - ts) > FRAME_SEC:
            continue
        if dist_m <= 0.02:
            continue

        rel = _rel_deg(theta, front_abs)
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

def _sector_stats(front_points: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = {
        "LEFT":   {"count": 0.0, "min_dist": float("inf")},
        "CENTER": {"count": 0.0, "min_dist": float("inf")},
        "RIGHT":  {"count": 0.0, "min_dist": float("inf")},
    }
    for p in front_points:
        sec = p.get("k3_sector")
        if sec not in out:
            continue
        d = float(p["dist_m"])
        out[sec]["count"] += 1.0
        if d < out[sec]["min_dist"]:
            out[sec]["min_dist"] = d
    return out

def _min_front_any(front_points: List[Dict[str, Any]]) -> float:
    best = float("inf")
    for p in front_points:
        d = float(p["dist_m"])
        if d < best:
            best = d
    return best


# =======================
# Decision logic (Front-180 only)
# =======================
def _compute_decision() -> Dict[str, Any]:
    now = time.time()
    front_pts, last_ts = _collect_front_points(limit=2400)

    if (not front_pts) or ((now - last_ts) > 2.0):
        _avoid_set(False)
        return {"ok": False, "label": "STOP", "reason": "no_recent_front180_points", "ts": now}

    secs = _sector_stats(front_pts)
    min_front = _min_front_any(front_pts)

    Lmin = float(secs["LEFT"]["min_dist"])
    Cmin = float(secs["CENTER"]["min_dist"])
    Rmin = float(secs["RIGHT"]["min_dist"])

    Lcnt = int(secs["LEFT"]["count"])
    Ccnt = int(secs["CENTER"]["count"])
    Rcnt = int(secs["RIGHT"]["count"])

    # Emergency STOP one-shot
    if (min_front <= EMERGENCY_STOP_M) and _em_is_armed():
        _em_disarm_after_issue()
        _avoid_set(True)
        return {
            "ok": True,
            "label": "STOP",
            "reason": f"emergency_stop_once(min_front={min_front:.3f})",
            "ts": now,
            "debug": _debug_payload(secs, min_front, avoid_active=True, clear_streak=0)
        }

    # re-arm emergency when safe
    if (min_front > EMERGENCY_REARM_M) and (not _em_is_armed()):
        _em_arm()

    # ===== IMPORTANT FIX =====
    # Sector thiếu điểm => coi như "unknown" => treat as near (không cho GO_STRAIGHT tự tin)
    def _near_or_unknown(d: float, cnt: int) -> bool:
        if cnt < MIN_SECTOR_POINTS:
            return True
        return math.isfinite(d) and (d <= OBSTACLE_NEAR_M)

    near_any = (
        _near_or_unknown(Lmin, Lcnt) or
        _near_or_unknown(Cmin, Ccnt) or
        _near_or_unknown(Rmin, Rcnt)
    )

    avoid_active, streak = _avoid_get()

    if avoid_active:
        clear_now = (Cmin > CLEAR_GO_M) and (min_front > CLEAR_GO_M) and (not near_any)
        if clear_now:
            _avoid_clear_streak_inc()
        else:
            _avoid_clear_streak_reset()

        avoid_active2, streak2 = _avoid_get()
        if streak2 >= CLEAR_STREAK_N:
            _avoid_set(False)
            return {
                "ok": True,
                "label": "GO_STRAIGHT",
                "reason": "avoid_clear_confirmed -> go_straight",
                "ts": now,
                "debug": _debug_payload(secs, min_front, avoid_active=False, clear_streak=streak2)
            }

        return {
            "ok": True,
            "label": "TURN_RIGHT",
            "reason": "avoid_mode_turn_right_until_clear",
            "ts": now,
            "debug": _debug_payload(secs, min_front, avoid_active=True, clear_streak=streak2)
        }

    if near_any:
        _avoid_set(True)
        return {
            "ok": True,
            "label": "TURN_RIGHT",
            "reason": "obstacle_near_or_unknown -> start_avoid_turn_right",
            "ts": now,
            "debug": _debug_payload(secs, min_front, avoid_active=True, clear_streak=0)
        }

    return {
        "ok": True,
        "label": "GO_STRAIGHT",
        "reason": "front180_clear -> go_straight",
        "ts": now,
        "debug": _debug_payload(secs, min_front, avoid_active=False, clear_streak=0)
    }

def _debug_payload(secs: Dict[str, Dict[str, float]], min_front: float, avoid_active: bool, clear_streak: int) -> Dict[str, Any]:
    Lmin = float(secs["LEFT"]["min_dist"])
    Cmin = float(secs["CENTER"]["min_dist"])
    Rmin = float(secs["RIGHT"]["min_dist"])
    return {
        "front_only": True,
        "front_center_deg_used": _get_front_center_deg(),
        "front_mirror": _get_front_mirror(),
        "front_flip": _get_front_flip(),
        "frame_sec": float(FRAME_SEC),
        "min_sector_points": int(MIN_SECTOR_POINTS),
        "thresholds": {
            "obstacle_near_m": float(OBSTACLE_NEAR_M),
            "clear_go_m": float(CLEAR_GO_M),
            "clear_streak_n": int(CLEAR_STREAK_N),
            "emergency_stop_m": float(EMERGENCY_STOP_M),
            "emergency_rearm_m": float(EMERGENCY_REARM_M),
        },
        "sectors": {
            "LEFT": {"count": int(secs["LEFT"]["count"]), "min_dist": Lmin},
            "CENTER": {"count": int(secs["CENTER"]["count"]), "min_dist": Cmin},
            "RIGHT": {"count": int(secs["RIGHT"]["count"]), "min_dist": Rmin},
        },
        "min_front": float(min_front),
        "avoid": {"active": bool(avoid_active), "clear_streak": int(clear_streak)},
        "emergency": {"armed": _em_is_armed(), "issued_this_event": _em_was_issued_this_event()},
    }


# =======================
# Rendering: local front-180 map
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

    # front semicircle outline (forward = up)
    draw.arc(bbox, start=0, end=180, fill=(160, 160, 160), width=3)

    # robot
    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(20, 20, 20))

    def rel_to_px(rel_deg: float, dist_m: float) -> Tuple[int, int]:
        r = math.radians(rel_deg)
        x_fwd = dist_m * math.cos(r)
        y_left = dist_m * math.sin(r)
        px = int(cx + y_left * ppm)
        py = int(cy - x_fwd * ppm)
        return px, py

    # sector overlay
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

    # points
    for p in front_points:
        d = float(p["dist_m"])
        rel = float(p["rel_deg"])
        if d > VIEW_RANGE_M:
            continue
        px, py = rel_to_px(rel, d)
        if 0 <= px < W and 0 <= py < H:
            img.putpixel((px, py), (80, 80, 80))

    # action arrow
    if label == "GO_STRAIGHT":
        ax, ay = rel_to_px(0.0, min(1.0, VIEW_RANGE_M * 0.9))
        draw.line((cx, cy, ax, ay), fill=(0, 0, 0), width=4)
    elif label == "TURN_RIGHT":
        ax, ay = rel_to_px(-45.0, min(1.0, VIEW_RANGE_M * 0.9))
        draw.line((cx, cy, ax, ay), fill=(0, 0, 0), width=4)
    elif label == "TURN_LEFT":
        ax, ay = rel_to_px(+45.0, min(1.0, VIEW_RANGE_M * 0.9))
        draw.line((cx, cy, ax, ay), fill=(0, 0, 0), width=4)

    # HUD
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = None
        font_small = None

    dbg = decision.get("debug", {}) if isinstance(decision, dict) else {}
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

    hud = [
        f"label: {label}",
        f"front_center_deg={_get_front_center_deg():.1f} mirror={_get_front_mirror()} flip={_get_front_flip()}",
        f"frame_sec={FRAME_SEC:.2f} min_sector_pts={MIN_SECTOR_POINTS}",
        f"front180 | obstacle_near={OBSTACLE_NEAR_M:.2f}m clear_go={CLEAR_GO_M:.2f}m",
        f"L={fmt_dist(dL)} C={fmt_dist(dC)} R={fmt_dist(dR)}",
        f"reason: {str(decision.get('reason',''))[:80]}",
    ]
    y0 = 10
    for s in hud:
        draw.text((10, y0), s, fill=(0, 0, 0), font=font)
        y0 += 22

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =======================
# Points payload (front 180 only)
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
            "front_center_deg_used": float(_get_front_center_deg()),
            "front_half_deg": float(FRONT_HALF_DEG),
            "front_mirror": int(_get_front_mirror()),
            "front_flip": int(_get_front_flip()),
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

    avoid_active, streak = _avoid_get()

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

        "front_center_deg_used": float(_get_front_center_deg()),
        "front_mirror": int(_get_front_mirror()),
        "front_flip": int(_get_front_flip()),
        "front_half_deg": float(FRONT_HALF_DEG),
        "frame_sec": float(FRAME_SEC),
        "front_only": True,

        "avoid": {"active": bool(avoid_active), "clear_streak": int(streak)},
        "emergency": {
            "armed": _em_is_armed(),
            "issued_this_event": _em_was_issued_this_event(),
            "stop_m": float(EMERGENCY_STOP_M),
            "rearm_m": float(EMERGENCY_REARM_M),
        },

        "thresholds": {
            "obstacle_near_m": float(OBSTACLE_NEAR_M),
            "clear_go_m": float(CLEAR_GO_M),
            "clear_streak_n": int(CLEAR_STREAK_N),
            "min_sector_points": int(MIN_SECTOR_POINTS),
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
        front_pts, _ = _collect_front_points(limit=2400)
        png = _render_map_png(decision=dec, front_points=front_pts)
        with cache_lock:
            _map_cache["png"] = png
            _map_cache["ts"] = time.time()
    return Response(png, mimetype="image/png")

@app.get("/api/set_front_center")
def api_set_front_center():
    if LOCK_FRONT_CENTER:
        return jsonify({"ok": False, "err": "front_center is locked"}), 403
    deg = request.args.get("deg", type=float, default=None)
    if deg is None:
        return jsonify({"ok": False, "err": "missing deg"}), 400
    _set_front_center_deg(deg)
    return jsonify({"ok": True, "front_center_deg": float(_get_front_center_deg())})

@app.get("/api/set_mirror")
def api_set_mirror():
    v = request.args.get("v", type=int, default=None)
    if v is None:
        return jsonify({"ok": False, "err": "missing v"}), 400
    _set_front_mirror(v)
    return jsonify({"ok": True, "front_mirror": int(_get_front_mirror())})

@app.get("/api/set_flip")
def api_set_flip():
    v = request.args.get("v", type=int, default=None)
    if v is None:
        return jsonify({"ok": False, "err": "missing v"}), 400
    _set_front_flip(v)
    return jsonify({"ok": True, "front_flip": int(_get_front_flip())})

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

@app.post("/api/reset_logic")
def api_reset_logic():
    _avoid_set(False)
    _em_arm()
    return jsonify({"ok": True})

@app.get("/dashboard")
def dashboard():
    html = f"""
    <html>
      <head>
        <title>LiDAR Front-180 Map</title>
        <style>
          body {{ font-family: Arial; margin: 12px; }}
          .row {{ display:flex; gap:12px; align-items:flex-start; }}
          img {{ border:1px solid #ccc; border-radius:8px; }}
          .box {{ padding:10px; border:1px solid #ddd; border-radius:8px; min-width: 380px; }}
          .mono {{ font-family: monospace; white-space: pre; }}
          input {{ width:120px; }}
          button {{ padding:6px 10px; }}
          .btnrow {{ display:flex; gap:8px; align-items:center; flex-wrap: wrap; }}
        </style>
      </head>
      <body>
        <h3>LiDAR Front-180 Map + k=3 sectors (TURN_RIGHT until clear)</h3>
        <div class="row">
          <img id="map" src="/api/map.png?ts={time.time()}" width="{VIEW_SIZE_PX}" height="{VIEW_SIZE_PX}"/>
          <div class="box">
            <div class="btnrow">
              <b>Front Center (deg)</b>
              <input id="fc" value="{_get_front_center_deg():.1f}" />
              <button onclick="setFC()">Set</button>
              <button onclick="toggleFlip()">Flip Front/Back</button>
              <button onclick="toggleMirror()">Toggle Mirror</button>
            </div>

            <div style="margin-top:8px;"><b>Decision</b></div>
            <div id="label" class="mono">loading...</div>
            <hr/>
            <div><b>Status</b></div>
            <div id="status" class="mono">loading...</div>
          </div>
        </div>

        <script>
          async function setFC() {{
            try {{
              const v = parseFloat(document.getElementById('fc').value);
              await fetch('/api/set_front_center?deg=' + encodeURIComponent(v));
            }} catch(e) {{}}
          }}

          async function toggleFlip() {{
            try {{
              const s = await fetch('/api/status').then(r=>r.json());
              const next = (s.front_flip ? 0 : 1);
              await fetch('/api/set_flip?v=' + next);
            }} catch(e) {{}}
          }}

          async function toggleMirror() {{
            try {{
              const s = await fetch('/api/status').then(r=>r.json());
              const next = (s.front_mirror ? 0 : 1);
              await fetch('/api/set_mirror?v=' + next);
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
                  front: {{
                    center: s.front_center_deg_used,
                    mirror: s.front_mirror,
                    flip: s.front_flip,
                    frame_sec: s.frame_sec
                  }}
                }}, null, 2);

              document.getElementById('status').textContent =
                JSON.stringify({{
                  running: s.running,
                  age_s: s.age_s,
                  front_center: s.front_center_deg_used,
                  mirror: s.front_mirror,
                  flip: s.front_flip,
                  frame_sec: s.frame_sec,
                  thresholds: s.thresholds
                }}, null, 2);

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

@app.get("/")
def home():
    return Response(
        "<h3>lidarhub_front180 running</h3>"
        "<ul>"
        "<li><a href='/dashboard'>/dashboard</a></li>"
        "<li>/api/map.png</li>"
        "<li>/take_lidar_data</li>"
        "<li>/ask_lidar_decision</li>"
        "<li>/api/decision_label</li>"
        "<li>/api/decision</li>"
        "<li>/api/status</li>"
        "<li>/api/set_front_center?deg=...</li>"
        "<li>/api/set_flip?v=0|1</li>"
        "<li>/api/set_mirror?v=0|1</li>"
        "<li>/api/restart (POST)</li>"
        "<li>/api/reset_logic (POST)</li>"
        "</ul>",
        mimetype="text/html"
    )


# =======================
# Main
# =======================
def main():
    _set_front_center_deg(FRONT_CENTER_DEG)
    _set_front_mirror(FRONT_MIRROR)
    _set_front_flip(FRONT_FLIP)

    threading.Thread(target=lidar_thread_main, daemon=True).start()
    threading.Thread(target=points_worker, daemon=True).start()
    threading.Thread(target=decision_worker, daemon=True).start()
    threading.Thread(target=map_worker, daemon=True).start()

    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
