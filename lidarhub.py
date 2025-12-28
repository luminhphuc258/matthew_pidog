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

# Robust distance: average K điểm gần nhất
K_NEAR = int(os.environ.get("K_NEAR", "6"))

# Sector đủ điểm mới tin (giảm nhảy do thiếu điểm)
MIN_SECTOR_POINTS = int(os.environ.get("MIN_SECTOR_POINTS", "20"))

# ===== Thresholds =====
CLEAR_GO_M = float(os.environ.get("CLEAR_GO_M", "0.95"))  # clear để đi thẳng
CLEAR_STREAK_N = int(os.environ.get("CLEAR_STREAK_N", "2"))

# Emergency STOP one-shot
EMERGENCY_STOP_M = float(os.environ.get("EMERGENCY_STOP_M", "0.25"))
EMERGENCY_REARM_M = float(os.environ.get("EMERGENCY_REARM_M", "0.45"))

# Debounce label (chống nhảy label)
LABEL_CONFIRM_N = int(os.environ.get("LABEL_CONFIRM_N", "2"))

# ===== Avoid timeout (tránh bị kẹt turn vô hạn) =====
AVOID_MAX_SEC = float(os.environ.get("AVOID_MAX_SEC", "4.0"))

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

# ===== Avoid state (CHỈ bật sau STOP) =====
_avoid_lock = threading.Lock()
_avoid_active: bool = False
_avoid_clear_streak: int = 0
_avoid_dir: str = "TURN_RIGHT"  # TURN_RIGHT / TURN_LEFT
_avoid_start_ts: float = 0.0

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
# Reset logic (FIX kẹt turn)
# =======================
def _reset_all_logic():
    """
    Reset về "như từ đầu" để robot tính lại fresh:
    - tắt avoid + reset streak + reset dir default
    - re-arm emergency gate
    - reset label debounce
    """
    global _avoid_active, _avoid_clear_streak, _avoid_dir, _avoid_start_ts
    global _em_stop_armed, _em_stop_issued_this_event
    global _out_label, _pending_label, _pending_n

    with _avoid_lock:
        _avoid_active = False
        _avoid_clear_streak = 0
        _avoid_dir = "TURN_RIGHT"
        _avoid_start_ts = 0.0

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
    global _avoid_active, _avoid_clear_streak, _avoid_dir, _avoid_start_ts
    with _avoid_lock:
        _avoid_active = bool(active)
        if direction in ("TURN_RIGHT", "TURN_LEFT"):
            _avoid_dir = direction
        if active:
            if _avoid_start_ts <= 0.0:
                _avoid_start_ts = time.time()
        else:
            _avoid_clear_streak = 0
            _avoid_start_ts = 0.0
            _avoid_dir = "TURN_RIGHT"

def _avoid_get() -> Tuple[bool, int, str, float]:
    with _avoid_lock:
        return bool(_avoid_active), int(_avoid_clear_streak), str(_avoid_dir), float(_avoid_start_ts)

def _avoid_streak_inc():
    global _avoid_clear_streak
    with _avoid_lock:
        _avoid_clear_streak += 1

def _avoid_streak_reset():
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

def _is_clear_forward(C: Dict[str, Any], min_front: float) -> bool:
    # Bình thường: cần CENTER đủ điểm để tin
    if _unknown(C):
        return False
    return (float(C["kmean_near"]) > float(CLEAR_GO_M)) and (float(min_front) > float(CLEAR_GO_M))

def _is_clear_forward_relaxed(C: Dict[str, Any], min_front: float) -> bool:
    """
    FIX kẹt turn:
    Trong avoid mode, nếu min_front đã > CLEAR_GO_M thì coi như clear,
    KHÔNG bị kẹt vì thiếu điểm CENTER.
    """
    if float(min_front) <= float(CLEAR_GO_M):
        return False
    # nếu CENTER có đủ điểm thì check thêm kmean; còn không thì cho qua
    if not _unknown(C):
        return float(C["kmean_near"]) > float(CLEAR_GO_M)
    return True

def _choose_turn_dir(L: Dict[str, Any], R: Dict[str, Any]) -> str:
    # chọn hướng thoáng hơn dựa trên kmean_near (unknown coi là xấu)
    l_ok = not _unknown(L)
    r_ok = not _unknown(R)

    l_val = float(L["kmean_near"]) if l_ok else -1.0
    r_val = float(R["kmean_near"]) if r_ok else -1.0

    if l_val < 0 and r_val < 0:
        return "TURN_RIGHT"
    return "TURN_LEFT" if l_val > r_val else "TURN_RIGHT"


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
# Decision logic
# - STOP (one-shot) -> enter avoid (turn L/R)
# - Avoid: nếu clear -> RESET ALL LOGIC rồi GO_STRAIGHT
# - Không còn kẹt TURN do thiếu điểm CENTER
# =======================
def _pack_decision(label: str, reason: str, secs: Dict[str, Dict[str, Any]], min_front: float,
                   avoid_active: bool, clear_streak: int, avoid_dir: str, avoid_age: float) -> Dict[str, Any]:
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
            },
            "emergency": {"armed": _em_is_armed(), "issued_this_event": _em_was_issued_this_event()},
            "thresholds": {
                "clear_go_m": float(CLEAR_GO_M),
                "clear_streak_n": int(CLEAR_STREAK_N),
                "min_sector_points": int(MIN_SECTOR_POINTS),
                "k_near": int(K_NEAR),
                "label_confirm_n": int(LABEL_CONFIRM_N),
                "emergency_stop_m": float(EMERGENCY_STOP_M),
                "emergency_rearm_m": float(EMERGENCY_REARM_M),
            }
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

    L = secs["LEFT"]
    C = secs["CENTER"]
    R = secs["RIGHT"]

    # 1) Emergency STOP one-shot
    if (min_front <= float(EMERGENCY_STOP_M)) and _em_is_armed():
        _em_disarm_after_issue()
        turn_dir = _choose_turn_dir(L, R)
        _avoid_set(True, turn_dir)

        avoid_active, streak, avoid_dir, start_ts = _avoid_get()
        avoid_age = (time.time() - start_ts) if start_ts > 0 else 0.0

        raw = _pack_decision(
            label="STOP",
            reason=f"emergency_stop_once(min_front={min_front:.3f}) -> enter_avoid({turn_dir})",
            secs=secs,
            min_front=min_front,
            avoid_active=True,
            clear_streak=0,
            avoid_dir=turn_dir,
            avoid_age=avoid_age,
        )
        raw["label"] = _smooth_label(raw["label"])
        return raw

    # Re-arm emergency when safe again
    if (min_front > float(EMERGENCY_REARM_M)) and (not _em_is_armed()):
        _em_arm()

    # 2) Avoid mode (chỉ có thể xảy ra sau STOP)
    avoid_active, streak, avoid_dir, start_ts = _avoid_get()
    avoid_age = (time.time() - start_ts) if start_ts > 0 else 0.0

    if avoid_active:
        # timeout safety: tránh bị kẹt vô hạn
        if avoid_age >= float(AVOID_MAX_SEC):
            _reset_all_logic()
            # tính lại fresh ngay lập tức (fallthrough)
            avoid_active, streak, avoid_dir, start_ts = _avoid_get()
            avoid_age = 0.0

        clear_now = _is_clear_forward_relaxed(C, min_front)
        if clear_now:
            _avoid_streak_inc()
        else:
            _avoid_streak_reset()

        avoid_active2, streak2, avoid_dir2, start_ts2 = _avoid_get()
        avoid_age2 = (time.time() - start_ts2) if start_ts2 > 0 else 0.0

        if streak2 >= int(CLEAR_STREAK_N):
            # ✅ FIX chính: thoát avoid xong là RESET toàn bộ state để tính lại từ đầu
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
            )
            raw["label"] = _smooth_label(raw["label"])
            return raw

        # Continue turning theo hướng đã chọn
        raw = _pack_decision(
            label=avoid_dir2,
            reason="avoid_mode_turn_until_clear (triggered_by_stop)",
            secs=secs,
            min_front=min_front,
            avoid_active=True,
            clear_streak=streak2,
            avoid_dir=avoid_dir2,
            avoid_age=avoid_age2,
        )
        raw["label"] = _smooth_label(raw["label"])
        return raw

    # 3) Normal mode: ưu tiên GO_STRAIGHT nếu clear
    if _is_clear_forward(C, min_front):
        raw = _pack_decision(
            label="GO_STRAIGHT",
            reason="front_clear -> go_straight",
            secs=secs,
            min_front=min_front,
            avoid_active=False,
            clear_streak=0,
            avoid_dir="TURN_RIGHT",
            avoid_age=0.0,
        )
        raw["label"] = _smooth_label(raw["label"])
        return raw

    # 4) Không clear nhưng chưa emergency stop => theo policy của bạn vẫn GO_STRAIGHT
    raw = _pack_decision(
        label="GO_STRAIGHT",
        reason="not_clear_but_no_emergency_stop -> still_go_straight (per_policy)",
        secs=secs,
        min_front=min_front,
        avoid_active=False,
        clear_streak=0,
        avoid_dir="TURN_RIGHT",
        avoid_age=0.0,
    )
    raw["label"] = _smooth_label(raw["label"])
    return raw


# =======================
# Rendering
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

    def rel_to_px(rel_deg: float, dist_m: float) -> Tuple[int, int]:
        r = math.radians(rel_deg)
        x_fwd = dist_m * math.cos(r)
        y_left = dist_m * math.sin(r)
        px = int(cx + y_left * ppm)
        py = int(cy - x_fwd * ppm)
        return px, py

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

    avoid_dbg = dbg.get("avoid", {}) if isinstance(dbg, dict) else {}
    hud = [
        f"label: {label}",
        f"front_center={FRONT_CENTER_DEG_HARD:.1f} mirror={FRONT_MIRROR_HARD} flip={FRONT_FLIP_HARD}",
        f"frame_sec={FRAME_SEC:.2f} k_near={K_NEAR} confirm_n={LABEL_CONFIRM_N}",
        f"clear_go={CLEAR_GO_M:.2f}m em_stop={EMERGENCY_STOP_M:.2f}m avoid_max={AVOID_MAX_SEC:.1f}s",
        f"avoid: active={avoid_dbg.get('active')} dir={avoid_dbg.get('dir')} streak={avoid_dbg.get('clear_streak')} age={avoid_dbg.get('age_s')}",
        f"L={fmt_dist(dL)} C={fmt_dist(dC)} R={fmt_dist(dR)}",
        f"reason: {str(decision.get('reason',''))[:90]}",
    ]
    y0 = 10
    for s in hud:
        draw.text((10, y0), s, fill=(0, 0, 0), font=font)
        y0 += 22

    import io
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
# ROUTES
# =======================
@app.get("/api/status")
def api_status():
    with lock:
        pts_n = len(latest_points)
        ts = float(latest_ts)

    with decision_state_lock:
        dec = dict(latest_decision_full)
        lbl = latest_decision_label

    avoid_active, streak, avoid_dir, start_ts = _avoid_get()
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
            "clear_streak_n": int(CLEAR_STREAK_N),
            "min_sector_points": int(MIN_SECTOR_POINTS),
            "k_near": int(K_NEAR),
            "label_confirm_n": int(LABEL_CONFIRM_N),
            "avoid_max_sec": float(AVOID_MAX_SEC),
            "emergency_stop_m": float(EMERGENCY_STOP_M),
            "emergency_rearm_m": float(EMERGENCY_REARM_M),
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
        <title>LiDAR Front-180 Map</title>
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
        <h3>LiDAR Front-180 Map (STOP -> TURN, clear -> RESET STATE)</h3>
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
                  thresholds: (d.debug||{{}}).thresholds
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
        "<h3>lidarhub_front180_stop_then_turn_reset running</h3>"
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
