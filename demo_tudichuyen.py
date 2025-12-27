#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import socket
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import requests
from flask import Flask, jsonify, Response

from motion_controller import MotionController

# =========================
# CONFIG
# =========================
POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# Lidar server (localhost:9399)
LIDAR_BASE = "http://127.0.0.1:9399"
URL_LIDAR_STATUS        = f"{LIDAR_BASE}/api/status"
URL_LIDAR_DECISION_TXT  = f"{LIDAR_BASE}/api/decision_label"   # TEXT: STOP / GO_STRAIGHT / TURN_LEFT / TURN_RIGHT
URL_LIDAR_DATA          = f"{LIDAR_BASE}/take_lidar_data"      # JSON with points
URL_LIDAR_DECISION_FULL = f"{LIDAR_BASE}/ask_lidar_decision"   # JSON full (optional)

# Gesture + camera decision (localhost:8000)
GEST_BASE = "http://127.0.0.1:8000"
URL_GESTURE         = f"{GEST_BASE}/take_gesture_meaning"
URL_CAMERA_DECISION = f"{GEST_BASE}/take_camera_decision"

# Map web
MAP_PORT = 5000

# Loop
LOOP_HZ = 20.0
HTTP_TIMEOUT = 0.6

# Command refresh (reduce jitter)
CMD_REFRESH_SEC = 0.35  # resend same cmd at most every 0.35s

# -------------------------
# SAFETY + PREDICTION
# -------------------------
# Rule 1: if FRONT < 30cm => MUST turn immediately (no forward)
FRONT_IMMEDIATE_TURN_CM = float(os.environ.get("FRONT_IMMEDIATE_TURN_CM", "50.0"))

# Emergency stop (hard)
EMERGENCY_STOP_CM = float(os.environ.get("EMERGENCY_STOP_CM", "30.0"))

# Robot size (updated): width ~ 10cm (to pass narrow corridors)
ROBOT_WIDTH_CM = float(os.environ.get("ROBOT_WIDTH_CM", "12.0"))
CORRIDOR_MARGIN_CM = float(os.environ.get("CORRIDOR_MARGIN_CM", "2.0"))
LOOKAHEAD_CM = float(os.environ.get("LOOKAHEAD_CM", "120.0"))

# Prediction: "if move 2 seconds more, will collide?"
PREDICT_T_SEC = float(os.environ.get("PREDICT_T_SEC", "2.0"))
ROBOT_SPEED_CMPS = float(os.environ.get("ROBOT_SPEED_CMPS", "35.0"))  # ~0.35 m/s
SAFETY_MARGIN_CM = float(os.environ.get("SAFETY_MARGIN_CM", "5.0"))

def required_forward_cm() -> float:
    return ROBOT_SPEED_CMPS * PREDICT_T_SEC + SAFETY_MARGIN_CM

# For turning acceptance
SAFE_TURN_CM = float(os.environ.get("SAFE_TURN_CM", "25.0"))  # smaller because robot narrower now
SAFE_FORWARD_MIN_NOW_CM = float(os.environ.get("SAFE_FORWARD_MIN_NOW_CM", "35.0"))  # minimal to even try forward

# Gesture TTL
GESTURE_TTL_SEC = 1.2

# Debug
DBG_PRINT_EVERY_SEC = 1.0

# Turn hold: reduce from 2s -> 1s (recalc faster)
TURN_HOLD_SEC = float(os.environ.get("TURN_HOLD_SEC", "1.0"))

# =========================
# FACE UDP (optional)
# =========================
FACE_HOST = "127.0.0.1"
FACE_PORT = 39393
_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def set_face(emo: str):
    try:
        _face_sock.sendto(str(emo).encode("utf-8"), (FACE_HOST, FACE_PORT))
    except Exception:
        pass

# =========================
# HTTP Utils (JSON + TEXT)
# =========================
def http_get(url: str, timeout: float = HTTP_TIMEOUT) -> Optional[requests.Response]:
    try:
        return requests.get(url, timeout=timeout)
    except Exception:
        return None

def http_get_json(url: str, timeout: float = HTTP_TIMEOUT) -> Optional[dict]:
    r = http_get(url, timeout=timeout)
    if not r or r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

def http_get_text(url: str, timeout: float = HTTP_TIMEOUT) -> Optional[str]:
    r = http_get(url, timeout=timeout)
    if not r or r.status_code != 200:
        return None
    try:
        return (r.text or "").strip()
    except Exception:
        return None

# =========================
# Robot Status Manager
# =========================
class RobotState:
    def __init__(self):
        self.mode = "BOOT"   # BOOT / STAND / MOVE / TURN / STOP
        self.detail = ""
        self.ts = time.time()

    def set(self, mode: str, detail: str = ""):
        self.mode = mode
        self.detail = detail
        self.ts = time.time()

    def __repr__(self):
        return f"{self.mode}:{self.detail}" if self.detail else self.mode

# =========================
# Motion wrapper (NO BACK)
# =========================
class RobotMotion:
    """
    NO BACK movement:
      - Only STOP / FORWARD / TURN_LEFT / TURN_RIGHT
    """
    def __init__(self, motion: MotionController, state: RobotState):
        self.motion = motion
        self.state = state
        self.dog = getattr(motion, "dog", None)

    def set_led(self, color: str, bps: float = 0.5):
        try:
            if self.dog and hasattr(self.dog, "rgb_strip"):
                self.dog.rgb_strip.set_mode("breath", color, bps=bps)
        except Exception:
            pass

    def boot_stand(self):
        self.state.set("BOOT")
        self.motion.boot()
        try:
            self.motion.execute("STOP")
        except Exception:
            pass
        self.set_led("white", bps=0.35)
        self.state.set("STAND")

    def stop(self):
        try:
            self.motion.execute("STOP")
        except Exception:
            pass
        self.set_led("red", bps=0.4)
        self.state.set("STOP")

    def forward(self):
        self.set_led("blue", bps=0.6)
        self.state.set("MOVE", "FORWARD")
        try:
            self.motion.execute("FORWARD")
        except Exception:
            pass

    def turn_left(self):
        self.set_led("red", bps=0.55)
        self.state.set("TURN", "TURN_LEFT")
        try:
            self.motion.execute("TURN_LEFT")
        except Exception:
            pass

    def turn_right(self):
        self.set_led("red", bps=0.55)
        self.state.set("TURN", "TURN_RIGHT")
        try:
            self.motion.execute("TURN_RIGHT")
        except Exception:
            pass

# =========================
# Gesture Poller
# =========================
class GesturePoller:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_label = ""
        self.latest_ts = 0.0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            js = http_get_json(URL_GESTURE, timeout=0.4)
            if js and js.get("ok"):
                latest = js.get("latest", {}) or {}
                label = str(latest.get("label", "") or "").upper().strip()
                ts = float(latest.get("ts", 0.0) or 0.0)
                with self.lock:
                    if label:
                        self.latest_label = label
                        self.latest_ts = ts
            time.sleep(0.10)

    def get_active(self) -> Optional[str]:
        with self.lock:
            if not self.latest_label:
                return None
            if (time.time() - float(self.latest_ts)) > GESTURE_TTL_SEC:
                return None
            return self.latest_label

# =========================
# ACCEL / COLLISION DETECTOR
# =========================
class CollisionDetector:
    """
    Tries to detect collision using:
      1) Optional HTTP endpoint (ACCEL_URL env)
      2) Optional robot IMU if available (best-effort)
    If detected -> collision_recent() True for a short TTL.
    """
    def __init__(self, motion: MotionController):
        self.motion = motion
        self.dog = getattr(motion, "dog", None)

        self.accel_url = os.environ.get("ACCEL_URL", "").strip()
        self.poll_hz = float(os.environ.get("ACCEL_POLL_HZ", "30.0"))
        self.ttl_sec = float(os.environ.get("COLLISION_TTL_SEC", "0.8"))

        # thresholds (tune if needed)
        self.jerk_th = float(os.environ.get("COLLISION_JERK_TH", "6.0"))    # (g/s) rough
        self.mag_th  = float(os.environ.get("COLLISION_MAG_TH", "2.2"))     # (g) rough

        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.last_collision_ts = 0.0
        self.last_mag = None
        self.last_t = None
        self.last_axayaz = (0.0, 0.0, 0.0)

        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    def _read_accel_http(self) -> Optional[Tuple[float, float, float]]:
        if not self.accel_url:
            return None
        js = http_get_json(self.accel_url, timeout=0.35)
        if not isinstance(js, dict):
            return None

        # accept common key styles
        ax = js.get("ax", js.get("x", None))
        ay = js.get("ay", js.get("y", None))
        az = js.get("az", js.get("z", None))

        if ax is None and "accel" in js and isinstance(js["accel"], (list, tuple)) and len(js["accel"]) >= 3:
            ax, ay, az = js["accel"][0], js["accel"][1], js["accel"][2]

        try:
            ax = float(ax); ay = float(ay); az = float(az)
        except Exception:
            return None
        return (ax, ay, az)

    def _read_accel_imu(self) -> Optional[Tuple[float, float, float]]:
        """
        Best-effort for various IMU APIs (depends on your stack).
        """
        imu = None
        try:
            imu = getattr(self.dog, "imu", None) if self.dog else None
        except Exception:
            imu = None
        if not imu:
            return None

        # try common method names
        for fn in ("get_accel", "get_acceleration", "read_accel", "acceleration"):
            try:
                if hasattr(imu, fn):
                    v = getattr(imu, fn)()
                    if isinstance(v, (list, tuple)) and len(v) >= 3:
                        ax, ay, az = float(v[0]), float(v[1]), float(v[2])
                        return (ax, ay, az)
                    if isinstance(v, dict):
                        ax = float(v.get("x", v.get("ax", 0.0)))
                        ay = float(v.get("y", v.get("ay", 0.0)))
                        az = float(v.get("z", v.get("az", 0.0)))
                        return (ax, ay, az)
            except Exception:
                continue
        return None

    def _run(self):
        dt = 1.0 / max(5.0, self.poll_hz)
        while not self._stop.is_set():
            now = time.time()

            v = self._read_accel_http()
            if v is None:
                v = self._read_accel_imu()

            if v is not None:
                ax, ay, az = v
                mag = math.sqrt(ax*ax + ay*ay + az*az)

                with self.lock:
                    self.last_axayaz = (ax, ay, az)

                    if self.last_mag is not None and self.last_t is not None:
                        dtm = max(1e-3, now - self.last_t)
                        jerk = abs(mag - self.last_mag) / dtm

                        # collision trigger
                        if mag >= self.mag_th or jerk >= self.jerk_th:
                            self.last_collision_ts = now

                    self.last_mag = mag
                    self.last_t = now

            time.sleep(dt)

    def collision_recent(self) -> bool:
        with self.lock:
            return (time.time() - self.last_collision_ts) <= self.ttl_sec

    def debug(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "collision_recent": self.collision_recent(),
                "last_collision_age": (time.time() - self.last_collision_ts) if self.last_collision_ts else None,
                "ax": self.last_axayaz[0],
                "ay": self.last_axayaz[1],
                "az": self.last_axayaz[2],
            }

# =========================
# Lidar parsing helpers (tolerant)
# =========================
def _to_cm(dist_any: Any) -> Optional[float]:
    try:
        d = float(dist_any)
    except Exception:
        return None
    if d <= 0:
        return None
    if d < 20.0:
        return d * 100.0
    if d > 1000.0:
        return d / 10.0
    return d

def _extract_points(scan_json: dict) -> List[Tuple[float, float]]:
    if not isinstance(scan_json, dict):
        return []

    arr = None
    for key in ("points", "scan", "data"):
        if isinstance(scan_json.get(key, None), list):
            arr = scan_json.get(key)
            break

    pts: List[Tuple[float, float]] = []

    if isinstance(arr, list) and arr:
        for it in arr:
            if isinstance(it, dict):
                a = it.get("angle", it.get("deg", it.get("theta", it.get("a", None))))
                d = it.get("dist_cm", it.get("distance_cm", it.get("dist", it.get("r", it.get("d", None)))))
                if a is None or d is None:
                    continue
                try:
                    ang = float(a)
                except Exception:
                    continue
                dcm = _to_cm(d)
                if dcm is None:
                    continue
                pts.append((ang, dcm))
                continue

            if isinstance(it, (list, tuple)) and len(it) >= 2:
                a, d = it[0], it[1]
                try:
                    ang = float(a)
                except Exception:
                    continue
                dcm = _to_cm(d)
                if dcm is None:
                    continue
                pts.append((ang, dcm))
                continue

        if pts:
            return pts

    ang = scan_json.get("angles", None)
    dist = (
        scan_json.get("dists_cm", None)
        or scan_json.get("distances_cm", None)
        or scan_json.get("dists", None)
        or scan_json.get("distances", None)
    )
    if isinstance(ang, list) and isinstance(dist, list) and len(ang) == len(dist):
        for a, d in zip(ang, dist):
            try:
                angf = float(a)
            except Exception:
                continue
            dcm = _to_cm(d)
            if dcm is None:
                continue
            pts.append((angf, dcm))

    return pts

def _sector_min_distance(points: List[Tuple[float, float]], a1: float, a2: float) -> float:
    if not points:
        return 9999.0

    def norm(a):
        while a > 180: a -= 360
        while a < -180: a += 360
        return a

    a1n, a2n = norm(a1), norm(a2)
    mins: List[float] = []
    for ang, dist in points:
        an = norm(ang)
        d = float(dist)
        if d <= 0:
            continue
        if a1n <= a2n:
            if a1n <= an <= a2n:
                mins.append(d)
        else:
            if an >= a1n or an <= a2n:
                mins.append(d)
    return min(mins) if mins else 9999.0

def lidar_clearance(points: List[Tuple[float, float]]) -> Dict[str, float]:
    return {
        "FRONT": _sector_min_distance(points, -25, 25),
        "LEFT":  _sector_min_distance(points, 30, 110),
        "RIGHT": _sector_min_distance(points, -110, -30),
        "BACK":  min(_sector_min_distance(points, 140, 180), _sector_min_distance(points, -180, -140)),
    }

# =========================
# Corridor clearance (uses robot width = 10cm)
# =========================
def _wrap_deg(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def corridor_clearance_cm(points: List[Tuple[float, float]], heading_deg: float) -> float:
    """
    Compute min obstacle distance 'y' in a corridor centered on heading_deg:
      - corridor width = ROBOT_WIDTH_CM + 2*CORRIDOR_MARGIN_CM
      - consider points within 0..LOOKAHEAD_CM forward
    Return +inf if no obstacle in corridor.
    """
    if not points:
        return float("inf")

    half_w = (ROBOT_WIDTH_CM * 0.5) + CORRIDOR_MARGIN_CM
    best_y = float("inf")

    # point format: angle_deg where 0deg is forward, positive => left (per your map)
    for ang_deg, dist_cm in points:
        d = float(dist_cm)
        if d <= 0:
            continue
        if d > LOOKAHEAD_CM:
            continue

        # rotate frame by heading_deg: rel = ang - heading
        rel = _wrap_deg(float(ang_deg) - float(heading_deg))
        if rel > 180.0:
            rel -= 360.0

        rr = math.radians(rel)
        x = math.sin(rr) * d     # left/right
        y = math.cos(rr) * d     # forward

        if y <= 0:
            continue
        if y > LOOKAHEAD_CM:
            continue
        if abs(x) <= half_w:
            if y < best_y:
                best_y = y

    return best_y

def choose_turn_dir(points: List[Tuple[float, float]], clr: Dict[str, float]) -> str:
    """
    Decide TURN_LEFT or TURN_RIGHT using corridor clearance (+45/-45) first,
    fallback to sector clearance LEFT/RIGHT.
    """
    c_left = corridor_clearance_cm(points, +45.0)
    c_right = corridor_clearance_cm(points, -45.0)

    # if both inf (no points), fallback
    if (not math.isfinite(c_left)) and (not math.isfinite(c_right)):
        left = float(clr.get("LEFT", 0.0) or 0.0)
        right = float(clr.get("RIGHT", 0.0) or 0.0)
        return "TURN_LEFT" if left >= right else "TURN_RIGHT"

    # prefer bigger clearance
    if c_left >= c_right:
        return "TURN_LEFT"
    return "TURN_RIGHT"

# =========================
# Direction picker (NO BACK) + prediction
# =========================
def forward_will_collide(front_cm: float) -> bool:
    """
    True if going forward for PREDICT_T_SEC likely collides.
    """
    th = required_forward_cm()
    return float(front_cm) <= th

def pick_direction_no_back(points: List[Tuple[float, float]], clr: Dict[str, float], prefer: str = "FRONT") -> str:
    """
    Use:
      - immediate front rule handled outside
      - prediction (2s) for forward
      - corridor clearance with ROBOT_WIDTH_CM=10cm
    """
    front = float(clr.get("FRONT", 0.0) or 0.0)
    left  = float(clr.get("LEFT", 0.0) or 0.0)
    right = float(clr.get("RIGHT", 0.0) or 0.0)

    # emergency
    if min(front, left, right) < EMERGENCY_STOP_CM:
        return "STOP"

    # corridor check for forward
    c0 = corridor_clearance_cm(points, 0.0)
    can_forward_now = (front >= SAFE_FORWARD_MIN_NOW_CM) and (c0 >= SAFE_FORWARD_MIN_NOW_CM)
    safe_forward_pred = (not forward_will_collide(front)) and (c0 >= required_forward_cm())

    if prefer == "FRONT" and can_forward_now and safe_forward_pred:
        return "FORWARD"

    # Otherwise pick turn direction if side looks better
    # (still require minimal space to turn)
    if max(left, right) >= SAFE_TURN_CM:
        return "TURN_LEFT" if left >= right else "TURN_RIGHT"

    # if forward is OK short-term but fails prediction, still prefer turning
    if can_forward_now and not safe_forward_pred:
        # turn away from tighter side
        return "TURN_LEFT" if left >= right else "TURN_RIGHT"

    # if forward is both ok now + prediction, use it
    if can_forward_now and safe_forward_pred:
        return "FORWARD"

    return "STOP"

# =========================
# Camera Decision (robust)
# =========================
def camera_decision_raw() -> Optional[dict]:
    return http_get_json(URL_CAMERA_DECISION, timeout=0.55)

def camera_is_clear(js: Optional[dict]) -> bool:
    if not js:
        return True
    if js.get("ok") is False:
        return True
    label = str(js.get("label", "") or "").lower().strip()
    if "no obstacle" in label:
        return True
    if ("yes" in label and "obstacle" in label) or ("have obstacle" in label):
        return False
    return True

# =========================
# Lidar status / decision (TEXT + JSON)
# =========================
def lidar_ready() -> bool:
    js = http_get_json(URL_LIDAR_STATUS, timeout=0.6)
    if not js:
        return False
    for k in ("ok", "ready", "lidar_ready", "running"):
        if k in js:
            try:
                return bool(js[k])
            except Exception:
                pass
    st = str(js.get("status", "") or "").lower()
    return ("ready" in st) or ("running" in st)

def normalize_lidar_label(s: str) -> str:
    """
    Normalize label from lidarhub.
    We explicitly DISALLOW BACK.
    """
    t = (s or "").strip().upper()
    if t in ("GO_STRAIGHT", "STRAIGHT", "FORWARD"):
        return "FORWARD"
    if t in ("TURNLEFT",):
        return "TURN_LEFT"
    if t in ("TURNRIGHT",):
        return "TURN_RIGHT"
    if t in ("BACK", "BACKWARD", "GO_BACK"):
        return "STOP"
    if t not in ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"):
        return "STOP"
    return t

def lidar_decision_label_text() -> Tuple[Optional[str], Optional[str]]:
    raw = http_get_text(URL_LIDAR_DECISION_TXT, timeout=0.55)
    if not raw:
        return None, None
    return raw, normalize_lidar_label(raw)

def lidar_decision_full_json() -> Optional[dict]:
    return http_get_json(URL_LIDAR_DECISION_FULL, timeout=0.55)

def lidar_scan_points() -> Tuple[List[Tuple[float, float]], Optional[dict]]:
    js = http_get_json(URL_LIDAR_DATA, timeout=0.75)
    if not js:
        return [], None
    pts = _extract_points(js)
    return pts, js

# =========================
# Map Web (2D top-down)
# =========================
class MapServer:
    def __init__(self):
        self.app = Flask("demo_map_2d")
        self.lock = threading.Lock()

        self.ts = time.time()
        self.last_points: List[Tuple[float, float]] = []
        self.last_clearance: Dict[str, float] = {}
        self.last_cmd: str = "STOP"
        self.robot_state: str = "BOOT"

        self.last_lidar_dec_raw: str = ""
        self.last_lidar_dec_norm: str = ""
        self.last_cam_label: str = ""
        self.last_pts_n: int = 0

        self.last_pred_th: float = required_forward_cm()
        self.last_pred_collide: bool = False
        self.last_collision: bool = False
        self.last_reason: str = ""

        @self.app.get("/map.json")
        def map_json():
            with self.lock:
                payload = {
                    "ts": self.ts,
                    "points": [{"angle": a, "dist_cm": d} for a, d in self.last_points[:2500]],
                    "clearance": self.last_clearance,
                    "cmd": self.last_cmd,
                    "robot_state": self.robot_state,
                    "dbg": {
                        "pts_n": self.last_pts_n,
                        "lidar_dec_raw": self.last_lidar_dec_raw,
                        "lidar_dec_norm": self.last_lidar_dec_norm,
                        "cam_label": self.last_cam_label,
                        "pred_th_cm": self.last_pred_th,
                        "pred_collide": self.last_pred_collide,
                        "collision": self.last_collision,
                        "reason": self.last_reason,
                        "robot_width_cm": ROBOT_WIDTH_CM,
                    }
                }
            return jsonify(payload)

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

    def _html(self) -> str:
        html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Robot Lidar 2D Map</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }
    .bar { padding:10px 14px; background:#111827; position:sticky; top:0; border-bottom:1px solid #223; }
    .wrap { display:flex; gap:14px; padding:14px; }
    canvas { background:#05070a; border:1px solid #223; border-radius:10px; }
    .card { background:#0f172a; border:1px solid #223; border-radius:10px; padding:12px; min-width:340px; }
    .kv { margin:6px 0; }
    .k { color:#93c5fd; }
    .small { font-size: 12px; color:#aab; line-height: 1.35; }
  </style>
</head>
<body>
  <div class="bar">
    <b>2D Lidar Map</b> — refresh 5 fps — URL: http://&lt;pi_ip&gt;:{MAP_PORT}/
  </div>

  <div class="wrap">
    <canvas id="cv" width="860" height="560"></canvas>
    <div class="card">
      <div class="kv"><span class="k">Robot State:</span> <span id="st">-</span></div>
      <div class="kv"><span class="k">Cmd:</span> <span id="cmd">-</span></div>

      <div class="kv"><span class="k">FRONT:</span> <span id="f">-</span> cm</div>
      <div class="kv"><span class="k">LEFT:</span> <span id="l">-</span> cm</div>
      <div class="kv"><span class="k">RIGHT:</span> <span id="r">-</span> cm</div>
      <div class="kv"><span class="k">BACK:</span> <span id="b">-</span> cm</div>

      <hr style="border:0;border-top:1px solid #223;margin:10px 0;">

      <div class="kv"><span class="k">pred_th (2s):</span> <span id="pth">-</span> cm</div>
      <div class="kv"><span class="k">pred_collide:</span> <span id="pc">-</span></div>
      <div class="kv"><span class="k">collision(accel):</span> <span id="col">-</span></div>
      <div class="kv"><span class="k">reason:</span> <span id="rs">-</span></div>

      <hr style="border:0;border-top:1px solid #223;margin:10px 0;">

      <div class="kv"><span class="k">pts_n:</span> <span id="pn">-</span></div>
      <div class="kv"><span class="k">lidar_dec_raw:</span> <span id="ldr">-</span></div>
      <div class="kv"><span class="k">lidar_dec_norm:</span> <span id="ldn">-</span></div>
      <div class="kv"><span class="k">cam_label:</span> <span id="clb">-</span></div>
      <div class="kv"><span class="k">TS:</span> <span id="ts">-</span></div>

      <div class="small" style="margin-top:10px;">
        Tip: Robot ở giữa (center), hướng tiến lên là phía trên.<br/>
        Vòng tròn: 50cm/vòng. Màu đỏ = vật cản gần.
      </div>
    </div>
  </div>

<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');

function computeScalePxPerCm(maxCm) {
  const margin = 28;
  const radiusPx = Math.min(cv.width, cv.height) * 0.5 - margin;
  return radiusPx / maxCm;
}

function drawGrid(ox, oy, pxPerCm, maxCm) {
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#142033';
  ctx.fillStyle = '#334155';
  ctx.font = '12px Arial';

  for (let cm=50; cm<=maxCm; cm+=50) {
    const r = cm * pxPerCm;
    ctx.beginPath();
    ctx.arc(ox, oy, r, 0, Math.PI*2);
    ctx.stroke();
    ctx.fillText(cm + "cm", ox + r + 6, oy + 4);
  }

  ctx.strokeStyle = '#0f2a4a';
  ctx.beginPath(); ctx.moveTo(ox, 10); ctx.lineTo(ox, cv.height-10); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(10, oy); ctx.lineTo(cv.width-10, oy); ctx.stroke();
}

function drawRobotCenter(ox, oy) {
  ctx.fillStyle = '#60a5fa';
  ctx.beginPath(); ctx.arc(ox, oy, 6, 0, Math.PI*2); ctx.fill();

  ctx.strokeStyle = '#60a5fa';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(ox, oy);
  ctx.lineTo(ox, oy - 26);
  ctx.stroke();
}

function pointColorByDistance(d) {
  if (d <= 120) return 'rgb(239,68,68)';
  if (d <= 200) return 'rgb(248,113,113)';
  const b = Math.max(70, Math.min(210, 260 - d));
  return `rgb(${b},${b},${b})`;
}

function drawPoints(data, ox, oy, pxPerCm, maxCm) {
  const pts = data.points || [];
  for (const p of pts) {
    const ang = (p.angle || 0) * Math.PI/180.0;
    const d = (p.dist_cm || 0);
    if (!d || d <= 0) continue;
    if (d > maxCm) continue;

    const x = Math.sin(ang) * d;
    const y = Math.cos(ang) * d;

    const sx = ox + x * pxPerCm;
    const sy = oy - y * pxPerCm;

    ctx.fillStyle = pointColorByDistance(d);
    ctx.fillRect(sx, sy, 2, 2);
  }
}

function draw(data) {
  ctx.clearRect(0,0,cv.width,cv.height);

  const ox = cv.width * 0.5;
  const oy = cv.height * 0.5;
  const maxCm = 300;
  const pxPerCm = computeScalePxPerCm(maxCm);

  drawGrid(ox, oy, pxPerCm, maxCm);
  drawPoints(data, ox, oy, pxPerCm, maxCm);
  drawRobotCenter(ox, oy);

  document.getElementById('st').textContent = data.robot_state || '-';
  document.getElementById('cmd').textContent = data.cmd || '-';
  document.getElementById('ts').textContent = String(data.ts || '-');

  const c = data.clearance || {};
  document.getElementById('f').textContent = (c.FRONT ?? '-');
  document.getElementById('l').textContent = (c.LEFT ?? '-');
  document.getElementById('r').textContent = (c.RIGHT ?? '-');
  document.getElementById('b').textContent = (c.BACK ?? '-');

  const dbg = data.dbg || {};
  document.getElementById('pn').textContent = (dbg.pts_n ?? '-');
  document.getElementById('ldr').textContent = (dbg.lidar_dec_raw ?? '-');
  document.getElementById('ldn').textContent = (dbg.lidar_dec_norm ?? '-');
  document.getElementById('clb').textContent = (dbg.cam_label ?? '-');

  document.getElementById('pth').textContent = (dbg.pred_th_cm ?? '-');
  document.getElementById('pc').textContent = String(dbg.pred_collide ?? '-');
  document.getElementById('col').textContent = String(dbg.collision ?? '-');
  document.getElementById('rs').textContent = String(dbg.reason ?? '-');
}

async function tick() {
  try {
    const r = await fetch('/map.json', {cache:'no-store'});
    const data = await r.json();
    draw(data);
  } catch(e) {}
}

setInterval(tick, 200);
tick();
</script>
</body>
</html>
"""
        return html.replace("{MAP_PORT}", str(MAP_PORT))

    def update(
        self,
        points: List[Tuple[float, float]],
        clearance: Dict[str, float],
        cmd: str,
        robot_state: str,
        lidar_dec_raw: str = "",
        lidar_dec_norm: str = "",
        cam_label: str = "",
        pred_collide: bool = False,
        collision: bool = False,
        reason: str = ""
    ):
        with self.lock:
            self.last_points = points
            self.last_pts_n = len(points)
            self.last_clearance = clearance
            self.last_cmd = cmd
            self.robot_state = robot_state
            self.last_lidar_dec_raw = lidar_dec_raw or ""
            self.last_lidar_dec_norm = lidar_dec_norm or ""
            self.last_cam_label = cam_label or ""
            self.last_pred_th = required_forward_cm()
            self.last_pred_collide = bool(pred_collide)
            self.last_collision = bool(collision)
            self.last_reason = reason or ""
            self.ts = time.time()

    def _port_available(self, port: int) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            s.close()
            return True
        except Exception:
            return False

    def run_bg(self, port: int):
        if not self._port_available(port):
            print(f"[MAP] ERROR: port {port} is already in use. Web map will NOT start.", flush=True)
            return

        def _serve():
            try:
                self.app.run(
                    host="0.0.0.0",
                    port=port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            except Exception as e:
                print(f"[MAP] ERROR: failed to start Flask on port {port}: {e}", flush=True)

        threading.Thread(target=_serve, daemon=True).start()
        print(f"[MAP] started on 0.0.0.0:{port}", flush=True)

# =========================
# Cmd sender (rate limited)
# =========================
def send_cmd_rate_limited(
    rm: RobotMotion,
    cmd: str,
    last_sent_cmd: str,
    last_sent_ts: float
) -> Tuple[str, float]:
    now = time.time()
    need_send = (cmd != last_sent_cmd) or ((now - last_sent_ts) >= CMD_REFRESH_SEC)
    if not need_send:
        return last_sent_cmd, last_sent_ts

    if cmd == "FORWARD":
        rm.forward()
    elif cmd == "TURN_LEFT":
        rm.turn_left()
    elif cmd == "TURN_RIGHT":
        rm.turn_right()
    else:
        rm.stop()

    return cmd, now

# =========================
# TURN HOLD MANAGER (force 1s)
# =========================
class TurnHold:
    def __init__(self, hold_sec: float = 1.0):
        self.lock = threading.Lock()
        self.hold_sec = float(hold_sec)
        self.active_cmd: str = ""   # TURN_LEFT / TURN_RIGHT
        self.until_ts: float = 0.0

    def arm(self, cmd: str):
        if cmd not in ("TURN_LEFT", "TURN_RIGHT"):
            return
        now = time.time()
        with self.lock:
            self.active_cmd = cmd
            self.until_ts = now + self.hold_sec

    def get(self) -> Optional[str]:
        now = time.time()
        with self.lock:
            if not self.active_cmd:
                return None
            if now <= self.until_ts:
                return self.active_cmd
            self.active_cmd = ""
            self.until_ts = 0.0
            return None

    def clear(self):
        with self.lock:
            self.active_cmd = ""
            self.until_ts = 0.0

# =========================
# Main autopilot (NO BACK)
# =========================
def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    state = RobotState()
    motion = MotionController(pose_file=POSE_FILE)
    rm = RobotMotion(motion=motion, state=state)

    map_server = MapServer()
    map_server.run_bg(MAP_PORT)

    gp = GesturePoller()
    gp.start()

    # collision detector (accelerometer)
    coldet = CollisionDetector(motion)
    coldet.start()

    turn_hold = TurnHold(hold_sec=TURN_HOLD_SEC)

    rm.boot_stand()
    set_face("what_is_it")

    print("[DEMO] waiting lidar ready...", flush=True)
    t0 = time.time()
    while True:
        if lidar_ready():
            print("[DEMO] lidar ready ✅", flush=True)
            break
        if time.time() - t0 > 12.0:
            print("[DEMO] lidar not ready yet (still continue) ⚠️", flush=True)
            break
        time.sleep(0.25)

    dt = 1.0 / LOOP_HZ

    last_cmd = "STOP"
    last_sent_cmd = "STOP"
    last_sent_ts = 0.0
    last_dbg_ts = 0.0

    print(f"[DEMO] map web: http://<pi_ip>:{MAP_PORT}/", flush=True)
    print(f"[CFG] FRONT_IMMEDIATE_TURN_CM={FRONT_IMMEDIATE_TURN_CM}  "
          f"predict_th(2s)={required_forward_cm():.1f}cm  "
          f"ROBOT_WIDTH_CM={ROBOT_WIDTH_CM}  TURN_HOLD_SEC={TURN_HOLD_SEC}", flush=True)

    try:
        while True:
            # read sensors first
            pts, scan_js = lidar_scan_points()
            clr = lidar_clearance(pts)

            lidar_dec_raw, lidar_dec_norm = lidar_decision_label_text()

            cam_js = camera_decision_raw()
            cam_label = str((cam_js or {}).get("label", "") or "")
            cam_clear = camera_is_clear(cam_js)

            front_cm = float(clr.get("FRONT", 9999.0) or 9999.0)
            pred_collide = forward_will_collide(front_cm)

            collision_now = coldet.collision_recent()

            # holding turn (1s) to avoid jitter
            holding = turn_hold.get()

            reason = ""

            # ---------- 0) Collision (accelerometer) override ----------
            if collision_now:
                # stop and immediately turn away from tighter side
                turn_hold.clear()
                td = choose_turn_dir(pts, clr)
                last_cmd = td
                turn_hold.arm(td)
                reason = "collision_detected_accel -> turn"

            # ---------- 1) Hard emergency stop ----------
            elif front_cm < EMERGENCY_STOP_CM:
                turn_hold.clear()
                last_cmd = "STOP"
                reason = "emergency_stop(front<EMERGENCY_STOP_CM)"

            # ---------- 2) IMPORTANT BUGFIX: front <30cm => MUST turn now ----------
            elif front_cm < FRONT_IMMEDIATE_TURN_CM:
                td = choose_turn_dir(pts, clr)
                last_cmd = td
                turn_hold.arm(td)
                reason = f"front<{FRONT_IMMEDIATE_TURN_CM:.0f}cm -> immediate_turn"

            # ---------- 3) If holding TURN -> keep it (do not recalc) ----------
            elif holding:
                last_cmd = holding
                reason = f"turn_hold({TURN_HOLD_SEC:.1f}s)"

            else:
                # ---------- 4) Gesture override (NO BACK) ----------
                g = gp.get_active()
                if g:
                    g = g.upper().strip()
                    if g in ("STOP", "STOPMUSIC", "SIT", "STANDUP"):
                        last_cmd = "STOP"
                        reason = f"gesture:{g}"
                    elif g in ("MOVELEFT", "TURNLEFT"):
                        last_cmd = "TURN_LEFT"
                        turn_hold.arm("TURN_LEFT")
                        reason = f"gesture:{g}"
                    elif g in ("MOVERIGHT", "TURNRIGHT"):
                        last_cmd = "TURN_RIGHT"
                        turn_hold.arm("TURN_RIGHT")
                        reason = f"gesture:{g}"
                    else:
                        last_cmd = "STOP"
                        reason = f"gesture:unknown({g})"
                else:
                    # ---------- 5) Lidarhub decision primary (NO BACK) ----------
                    dec = (lidar_dec_norm or "FORWARD").strip().upper()
                    if dec not in ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"):
                        dec = "STOP"

                    # If lidar says forward but camera blocks OR prediction says collide -> turn immediately
                    if dec == "FORWARD":
                        if not cam_clear:
                            td = choose_turn_dir(pts, clr)
                            last_cmd = td
                            turn_hold.arm(td)
                            reason = "lidar:FORWARD but camera_block -> turn"
                        elif pred_collide:
                            td = choose_turn_dir(pts, clr)
                            last_cmd = td
                            turn_hold.arm(td)
                            reason = f"lidar:FORWARD but predict_collide(th={required_forward_cm():.1f}) -> turn"
                        else:
                            # corridor + forward safe
                            c0 = corridor_clearance_cm(pts, 0.0)
                            if front_cm >= SAFE_FORWARD_MIN_NOW_CM and c0 >= required_forward_cm():
                                last_cmd = "FORWARD"
                                reason = "forward_ok(now + predict + corridor)"
                            else:
                                td = choose_turn_dir(pts, clr)
                                last_cmd = td
                                turn_hold.arm(td)
                                reason = "forward_not_good_in_corridor -> turn"

                    elif dec in ("TURN_LEFT", "TURN_RIGHT"):
                        # allow but sanity check: if chosen side is super tight, flip to other side
                        left = float(clr.get("LEFT", 0.0) or 0.0)
                        right = float(clr.get("RIGHT", 0.0) or 0.0)
                        if dec == "TURN_LEFT" and left < SAFE_TURN_CM and right >= SAFE_TURN_CM:
                            last_cmd = "TURN_RIGHT"
                            turn_hold.arm("TURN_RIGHT")
                            reason = "lidar:TURN_LEFT but left_tight -> TURN_RIGHT"
                        elif dec == "TURN_RIGHT" and right < SAFE_TURN_CM and left >= SAFE_TURN_CM:
                            last_cmd = "TURN_LEFT"
                            turn_hold.arm("TURN_LEFT")
                            reason = "lidar:TURN_RIGHT but right_tight -> TURN_LEFT"
                        else:
                            last_cmd = dec
                            turn_hold.arm(dec)
                            reason = f"lidar:{dec}"

                    else:  # STOP
                        # try to find best safe move (no back), still using prediction
                        best = pick_direction_no_back(pts, clr, prefer="FRONT")
                        if best in ("TURN_LEFT", "TURN_RIGHT"):
                            turn_hold.arm(best)
                        last_cmd = best
                        reason = f"lidar:STOP -> pick:{best}"

            # Send cmd (rate-limited)
            last_sent_cmd, last_sent_ts = send_cmd_rate_limited(
                rm=rm,
                cmd=last_cmd,
                last_sent_cmd=last_sent_cmd,
                last_sent_ts=last_sent_ts
            )

            # Update map
            map_server.update(
                points=pts,
                clearance=clr,
                cmd=last_cmd,
                robot_state=str(state),
                lidar_dec_raw=lidar_dec_raw or "",
                lidar_dec_norm=lidar_dec_norm or "",
                cam_label=cam_label,
                pred_collide=pred_collide,
                collision=collision_now,
                reason=reason
            )

            # Debug log
            now = time.time()
            if (now - last_dbg_ts) >= DBG_PRINT_EVERY_SEC:
                last_dbg_ts = now
                scan_keys = list(scan_js.keys()) if isinstance(scan_js, dict) else []
                hold_info = holding if holding else "-"
                c0 = corridor_clearance_cm(pts, 0.0) if pts else float("inf")
                print(
                    f"[DBG] hold={hold_info} pts={len(pts)} "
                    f"front={front_cm:.1f} left={clr.get('LEFT',9999.0):.1f} right={clr.get('RIGHT',9999.0):.1f} "
                    f"c0={c0:.1f} pred_th={required_forward_cm():.1f} pred_collide={pred_collide} "
                    f"collision={collision_now} "
                    f"lidar_dec_raw={lidar_dec_raw} lidar_dec_norm={lidar_dec_norm} "
                    f"cam_label={cam_label!r} cam_clear={cam_clear} -> cmd={last_cmd} "
                    f"| reason={reason} | scan_keys={scan_keys}",
                    flush=True
                )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)

    finally:
        try:
            gp.stop()
        except Exception:
            pass
        try:
            coldet.stop()
        except Exception:
            pass
        try:
            rm.stop()
        except Exception:
            pass
        try:
            motion.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
