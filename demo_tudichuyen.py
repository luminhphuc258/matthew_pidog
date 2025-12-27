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
URL_LIDAR_DECISION_TXT  = f"{LIDAR_BASE}/api/decision_label"
URL_LIDAR_DATA          = f"{LIDAR_BASE}/take_lidar_data"
URL_LIDAR_DECISION_FULL = f"{LIDAR_BASE}/ask_lidar_decision"

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
CMD_REFRESH_SEC = 0.35

# -------------------------
# SAFETY + PREDICTION
# -------------------------
FRONT_IMMEDIATE_TURN_CM = float(os.environ.get("FRONT_IMMEDIATE_TURN_CM", "50.0"))
EMERGENCY_STOP_CM       = float(os.environ.get("EMERGENCY_STOP_CM", "30.0"))

ROBOT_WIDTH_CM      = float(os.environ.get("ROBOT_WIDTH_CM", "12.0"))
CORRIDOR_MARGIN_CM  = float(os.environ.get("CORRIDOR_MARGIN_CM", "2.0"))
LOOKAHEAD_CM        = float(os.environ.get("LOOKAHEAD_CM", "120.0"))

PREDICT_T_SEC       = float(os.environ.get("PREDICT_T_SEC", "2.0"))
ROBOT_SPEED_CMPS    = float(os.environ.get("ROBOT_SPEED_CMPS", "35.0"))
SAFETY_MARGIN_CM    = float(os.environ.get("SAFETY_MARGIN_CM", "5.0"))

SAFE_TURN_CM            = float(os.environ.get("SAFE_TURN_CM", "25.0"))
SAFE_FORWARD_MIN_NOW_CM = float(os.environ.get("SAFE_FORWARD_MIN_NOW_CM", "35.0"))

# ===== NEW: stuck recovery =====
ESCAPE_BACK_SEC      = float(os.environ.get("ESCAPE_BACK_SEC", "3.0"))
ESCAPE_MAX_TURN_SEC  = float(os.environ.get("ESCAPE_MAX_TURN_SEC", "6.0"))

# khi TURN lock: chỉ unlock khi FRONT >= ngưỡng này
TURN_UNLOCK_FRONT_CM = float(os.environ.get("TURN_UNLOCK_FRONT_CM", "80.0"))

# Gesture TTL
GESTURE_TTL_SEC = 1.2

# Debug
DBG_PRINT_EVERY_SEC = 1.0

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
# HTTP Utils
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
        self.mode = "BOOT"   # BOOT / STAND / MOVE / TURN / BACK / STOP
        self.detail = ""
        self.ts = time.time()

    def set(self, mode: str, detail: str = ""):
        self.mode = mode
        self.detail = detail
        self.ts = time.time()

    def __repr__(self):
        return f"{self.mode}:{self.detail}" if self.detail else self.mode

# =========================
# Motion wrapper (ALLOW BACK for escape)
# =========================
class RobotMotion:
    """
    Now supports STOP / FORWARD / TURN_LEFT / TURN_RIGHT / BACK
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

    def back(self):
        self.set_led("red", bps=0.65)
        self.state.set("BACK", "BACK")
        try:
            self.motion.execute("BACK")
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
# Lidar parsing helpers
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
# Corridor clearance
# =========================
def _wrap_deg(a: float) -> float:
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a

def corridor_clearance_cm(points: List[Tuple[float, float]], heading_deg: float) -> float:
    if not points:
        return float("inf")

    half_w = (ROBOT_WIDTH_CM * 0.5) + CORRIDOR_MARGIN_CM
    best_y = float("inf")

    for ang_deg, dist_cm in points:
        d = float(dist_cm)
        if d <= 0:
            continue
        if d > LOOKAHEAD_CM:
            continue

        rel = _wrap_deg(float(ang_deg) - float(heading_deg))
        if rel > 180.0:
            rel -= 360.0

        rr = math.radians(rel)
        x = math.sin(rr) * d
        y = math.cos(rr) * d

        if y <= 0:
            continue
        if y > LOOKAHEAD_CM:
            continue
        if abs(x) <= half_w:
            if y < best_y:
                best_y = y

    return best_y

def choose_turn_dir(points: List[Tuple[float, float]], clr: Dict[str, float]) -> str:
    c_left = corridor_clearance_cm(points, +45.0)
    c_right = corridor_clearance_cm(points, -45.0)

    if (not math.isfinite(c_left)) and (not math.isfinite(c_right)):
        left = float(clr.get("LEFT", 0.0) or 0.0)
        right = float(clr.get("RIGHT", 0.0) or 0.0)
        return "TURN_LEFT" if left >= right else "TURN_RIGHT"

    return "TURN_LEFT" if c_left >= c_right else "TURN_RIGHT"

# =========================
# Prediction
# =========================
def required_forward_cm() -> float:
    return ROBOT_SPEED_CMPS * PREDICT_T_SEC + SAFETY_MARGIN_CM

def forward_will_collide(front_cm: float) -> bool:
    return float(front_cm) <= required_forward_cm()

def pick_direction(points: List[Tuple[float, float]], clr: Dict[str, float]) -> str:
    front = float(clr.get("FRONT", 0.0) or 0.0)
    left  = float(clr.get("LEFT", 0.0) or 0.0)
    right = float(clr.get("RIGHT", 0.0) or 0.0)

    c0 = corridor_clearance_cm(points, 0.0)
    can_forward_now = (front >= SAFE_FORWARD_MIN_NOW_CM) and (c0 >= SAFE_FORWARD_MIN_NOW_CM)
    safe_forward_pred = (not forward_will_collide(front)) and (c0 >= required_forward_cm())

    if can_forward_now and safe_forward_pred:
        return "FORWARD"

    if max(left, right) >= SAFE_TURN_CM:
        return "TURN_LEFT" if left >= right else "TURN_RIGHT"

    return "STOP"

# =========================
# Camera
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
# Lidar status / decision
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
    t = (s or "").strip().upper()
    if t in ("GO_STRAIGHT", "STRAIGHT", "FORWARD"):
        return "FORWARD"
    if t in ("TURNLEFT",):
        return "TURN_LEFT"
    if t in ("TURNRIGHT",):
        return "TURN_RIGHT"
    # lidar BACK -> ignore (autopilot tự BACK khi cần)
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

def lidar_scan_points() -> Tuple[List[Tuple[float, float]], Optional[dict]]:
    js = http_get_json(URL_LIDAR_DATA, timeout=0.75)
    if not js:
        return [], None
    pts = _extract_points(js)
    return pts, js

# =========================
# Map Web
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
        self.last_reason: str = ""

        self.escape_state: str = "NONE"   # NONE / BACK / TURN
        self.lock_turn_cmd: str = ""      # TURN_LEFT / TURN_RIGHT

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
                        "reason": self.last_reason,
                        "robot_width_cm": ROBOT_WIDTH_CM,
                        "escape_state": self.escape_state,
                        "lock_turn_cmd": self.lock_turn_cmd,
                        "unlock_front_cm": TURN_UNLOCK_FRONT_CM,
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
      <div class="kv"><span class="k">reason:</span> <span id="rs">-</span></div>

      <hr style="border:0;border-top:1px solid #223;margin:10px 0;">

      <div class="kv"><span class="k">escape_state:</span> <span id="es">-</span></div>
      <div class="kv"><span class="k">lock_turn_cmd:</span> <span id="lt">-</span></div>

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
  document.getElementById('rs').textContent = String(dbg.reason ?? '-');

  document.getElementById('es').textContent = String(dbg.escape_state ?? '-');
  document.getElementById('lt').textContent = String(dbg.lock_turn_cmd ?? '-');
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
        reason: str = "",
        escape_state: str = "NONE",
        lock_turn_cmd: str = "",
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
            self.last_reason = reason or ""
            self.escape_state = escape_state
            self.lock_turn_cmd = lock_turn_cmd
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
                self.app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
            except Exception as e:
                print(f"[MAP] ERROR: failed to start Flask on port {port}: {e}", flush=True)

        threading.Thread(target=_serve, daemon=True).start()
        print(f"[MAP] started on 0.0.0.0:{port}", flush=True)

# =========================
# Cmd sender (rate limited)
# =========================
def send_cmd_rate_limited(rm: RobotMotion, cmd: str, last_sent_cmd: str, last_sent_ts: float) -> Tuple[str, float]:
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
    elif cmd == "BACK":
        rm.back()
    else:
        rm.stop()

    return cmd, now

# =========================
# TURN LOCK (keep one direction until front clear)
# =========================
class TurnLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.cmd: str = ""         # TURN_LEFT / TURN_RIGHT
        self.active: bool = False

    def arm(self, cmd: str):
        if cmd not in ("TURN_LEFT", "TURN_RIGHT"):
            return
        with self.lock:
            self.cmd = cmd
            self.active = True

    def clear(self):
        with self.lock:
            self.cmd = ""
            self.active = False

    def get(self) -> Optional[str]:
        with self.lock:
            if not self.active:
                return None
            return self.cmd

# =========================
# ESCAPE MANAGER (BACK 3s -> TURN locked until front clear)
# =========================
class EscapeManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "NONE"       # NONE / BACK / TURN
        self.until_ts = 0.0
        self.turn_cmd = ""
        self.turn_start_ts = 0.0

    def trigger(self, now: float, turn_cmd: str):
        with self.lock:
            self.state = "BACK"
            self.until_ts = now + ESCAPE_BACK_SEC
            self.turn_cmd = turn_cmd
            self.turn_start_ts = 0.0

    def set_turn(self, now: float, turn_cmd: str):
        with self.lock:
            self.state = "TURN"
            self.turn_cmd = turn_cmd
            self.turn_start_ts = now
            self.until_ts = now + ESCAPE_MAX_TURN_SEC  # timeout fail-safe

    def clear(self):
        with self.lock:
            self.state = "NONE"
            self.until_ts = 0.0
            self.turn_cmd = ""
            self.turn_start_ts = 0.0

    def snapshot(self) -> Tuple[str, float, str, float]:
        with self.lock:
            return self.state, self.until_ts, self.turn_cmd, self.turn_start_ts

# =========================
# Main
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

    turn_lock = TurnLock()
    escape = EscapeManager()

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
    print(f"[CFG] FRONT_IMMEDIATE_TURN_CM={FRONT_IMMEDIATE_TURN_CM}  EMERGENCY_STOP_CM={EMERGENCY_STOP_CM} "
          f"ESCAPE_BACK_SEC={ESCAPE_BACK_SEC} TURN_UNLOCK_FRONT_CM={TURN_UNLOCK_FRONT_CM}", flush=True)

    try:
        while True:
            pts, scan_js = lidar_scan_points()
            clr = lidar_clearance(pts)

            lidar_dec_raw, lidar_dec_norm = lidar_decision_label_text()

            cam_js = camera_decision_raw()
            cam_label = str((cam_js or {}).get("label", "") or "")
            cam_clear = camera_is_clear(cam_js)

            front_cm = float(clr.get("FRONT", 9999.0) or 9999.0)
            pred_collide = forward_will_collide(front_cm)

            now = time.time()
            reason = ""
            esc_state, esc_until, esc_turn_cmd, esc_turn_start = escape.snapshot()
            locked_turn = turn_lock.get()

            # ===== 0) Gesture override (STOP ưu tiên cao) =====
            g = gp.get_active()
            if g:
                g = g.upper().strip()
                if g in ("STOP", "STOPMUSIC", "SIT", "STANDUP"):
                    escape.clear()
                    turn_lock.clear()
                    last_cmd = "STOP"
                    reason = f"gesture:{g}"
                elif g in ("MOVELEFT", "TURNLEFT"):
                    escape.clear()
                    turn_lock.arm("TURN_LEFT")
                    last_cmd = "TURN_LEFT"
                    reason = f"gesture:{g} -> lock TURN_LEFT"
                elif g in ("MOVERIGHT", "TURNRIGHT"):
                    escape.clear()
                    turn_lock.arm("TURN_RIGHT")
                    last_cmd = "TURN_RIGHT"
                    reason = f"gesture:{g} -> lock TURN_RIGHT"
                else:
                    last_cmd = "STOP"
                    reason = f"gesture:unknown({g})"

            else:
                # ===== 1) ESCAPE state machine =====
                if esc_state == "BACK":
                    if now <= esc_until:
                        last_cmd = "BACK"
                        reason = f"ESCAPE:BACK({ESCAPE_BACK_SEC:.1f}s)"
                    else:
                        # back xong -> vào TURN theo hướng đã chọn
                        escape.set_turn(now, esc_turn_cmd or choose_turn_dir(pts, clr))
                        turn_lock.arm(escape.snapshot()[2])  # lock turn cmd
                        last_cmd = escape.snapshot()[2]
                        reason = f"ESCAPE:BACK done -> TURN {last_cmd}"

                elif esc_state == "TURN":
                    # giữ TURN cho tới khi FRONT clear mới thoát
                    if front_cm >= TURN_UNLOCK_FRONT_CM and cam_clear and (not pred_collide):
                        escape.clear()
                        turn_lock.clear()
                        last_cmd = "FORWARD"
                        reason = f"ESCAPE:TURN done (front>={TURN_UNLOCK_FRONT_CM:.0f}) -> FORWARD"
                    else:
                        # fail-safe timeout: quay quá lâu vẫn không clear -> back lại và chọn lại
                        if now >= esc_until:
                            td = choose_turn_dir(pts, clr)
                            escape.trigger(now, td)
                            turn_lock.clear()
                            last_cmd = "BACK"
                            reason = "ESCAPE:TURN timeout -> BACK again"
                        else:
                            last_cmd = esc_turn_cmd or "TURN_RIGHT"
                            reason = f"ESCAPE:TURN locked({last_cmd})"

                else:
                    # ===== 2) Nếu FRONT quá sát -> ESCAPE ngay (BACK 3s rồi TURN) =====
                    if front_cm < EMERGENCY_STOP_CM:
                        td = choose_turn_dir(pts, clr)
                        escape.trigger(now, td)
                        turn_lock.clear()
                        last_cmd = "BACK"
                        reason = f"front<{EMERGENCY_STOP_CM:.0f} -> ESCAPE:BACK then {td}"

                    else:
                        # ===== 3) TURN LOCK trong normal: đã quyết định quay => giữ tới khi front clear =====
                        if locked_turn:
                            if front_cm >= TURN_UNLOCK_FRONT_CM and cam_clear and (not pred_collide):
                                turn_lock.clear()
                                # sau khi unlock, tính lại bình thường
                                last_cmd = pick_direction(pts, clr)
                                reason = f"turn_lock cleared (front>={TURN_UNLOCK_FRONT_CM:.0f}) -> {last_cmd}"
                            else:
                                last_cmd = locked_turn
                                reason = f"turn_lock({locked_turn})"

                        else:
                            # ===== 4) Normal logic =====
                            dec = (lidar_dec_norm or "FORWARD").strip().upper()
                            if dec not in ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"):
                                dec = "STOP"

                            # front < immediate_turn -> lock turn ngay
                            if front_cm < FRONT_IMMEDIATE_TURN_CM:
                                td = choose_turn_dir(pts, clr)
                                turn_lock.arm(td)
                                last_cmd = td
                                reason = f"front<{FRONT_IMMEDIATE_TURN_CM:.0f} -> lock {td}"

                            else:
                                # lidar says forward but camera/predict blocks -> lock turn
                                if dec == "FORWARD":
                                    if (not cam_clear) or pred_collide:
                                        td = choose_turn_dir(pts, clr)
                                        turn_lock.arm(td)
                                        last_cmd = td
                                        reason = "lidar:FORWARD but blocked -> lock turn"
                                    else:
                                        best = pick_direction(pts, clr)
                                        if best in ("TURN_LEFT", "TURN_RIGHT"):
                                            turn_lock.arm(best)
                                            last_cmd = best
                                            reason = f"best={best} -> lock turn"
                                        else:
                                            last_cmd = best
                                            reason = f"best={best}"

                                elif dec in ("TURN_LEFT", "TURN_RIGHT"):
                                    turn_lock.arm(dec)
                                    last_cmd = dec
                                    reason = f"lidar:{dec} -> lock"

                                else:  # STOP
                                    best = pick_direction(pts, clr)
                                    if best in ("TURN_LEFT", "TURN_RIGHT"):
                                        turn_lock.arm(best)
                                        last_cmd = best
                                        reason = f"lidar:STOP -> pick {best} lock"
                                    else:
                                        # nếu STOP nhưng front vẫn OK, có thể là lidar jitter -> tự quyết
                                        last_cmd = best
                                        reason = f"lidar:STOP -> pick {best}"

            # Send cmd (rate-limited)
            last_sent_cmd, last_sent_ts = send_cmd_rate_limited(
                rm=rm,
                cmd=last_cmd,
                last_sent_cmd=last_sent_cmd,
                last_sent_ts=last_sent_ts
            )

            # Update map
            esc_state, esc_until, esc_turn_cmd, _ = escape.snapshot()
            map_server.update(
                points=pts,
                clearance=clr,
                cmd=last_cmd,
                robot_state=str(state),
                lidar_dec_raw=lidar_dec_raw or "",
                lidar_dec_norm=lidar_dec_norm or "",
                cam_label=cam_label,
                pred_collide=pred_collide,
                reason=reason,
                escape_state=esc_state,
                lock_turn_cmd=(turn_lock.get() or esc_turn_cmd or ""),
            )

            # Debug log
            if (now - last_dbg_ts) >= DBG_PRINT_EVERY_SEC:
                last_dbg_ts = now
                scan_keys = list(scan_js.keys()) if isinstance(scan_js, dict) else []
                c0 = corridor_clearance_cm(pts, 0.0) if pts else float("inf")
                print(
                    f"[DBG] esc={esc_state} lock={turn_lock.get() or '-'} "
                    f"front={front_cm:.1f} left={clr.get('LEFT',9999.0):.1f} right={clr.get('RIGHT',9999.0):.1f} "
                    f"c0={c0:.1f} pred_th={required_forward_cm():.1f} pred_collide={pred_collide} "
                    f"lidar_dec_raw={lidar_dec_raw} lidar_dec_norm={lidar_dec_norm} "
                    f"cam_label={cam_label!r} cam_clear={cam_clear} -> cmd={last_cmd} | reason={reason} "
                    f"| scan_keys={scan_keys}",
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
            rm.stop()
        except Exception:
            pass
        try:
            motion.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
