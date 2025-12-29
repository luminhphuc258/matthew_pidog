#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import socket
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import requests
from flask import Flask, jsonify, Response

from motion_controller import MotionController

# =========================
# CONFIG
# =========================
POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# LiDAR server (localhost:9399)
LIDAR_BASE = "http://127.0.0.1:9399"
URL_LIDAR_STATUS       = f"{LIDAR_BASE}/api/status"
URL_LIDAR_DECISION_TXT = f"{LIDAR_BASE}/api/decision_label"
URL_LIDAR_DATA         = f"{LIDAR_BASE}/take_lidar_data"

# Camera decision server (Gesture/WebDashboard port 8000)
CAM_BASE = os.environ.get("CAM_BASE", "http://127.0.0.1:8000")
URL_CAMERA_DECISION = f"{CAM_BASE}/take_camera_decision"

# Map web
MAP_PORT = 5000

# Loop
LOOP_HZ = 20.0
HTTP_TIMEOUT = 0.6

# Command refresh (reduce jitter)
CMD_REFRESH_SEC = 0.35

# Debug
DBG_PRINT_EVERY_SEC = 1.0

# ====== AVOIDANCE CONFIG ======
# Nếu LiDAR CLEAR nhưng camera thấy obstacle -> STOP + BACK + TURN
LIDAR_FRONT_CLEAR_CM = float(os.environ.get("LIDAR_FRONT_CLEAR_CM", "80"))  # front >= 80cm coi là khá clear
CAM_OBS_MIN_CONF     = float(os.environ.get("CAM_OBS_MIN_CONF", "0.35"))     # camera conf tối thiểu để tin
BACK_SEC             = float(os.environ.get("BACK_SEC", "1.0"))              # lùi lại bao lâu
TURN_OVERRIDE_SEC    = float(os.environ.get("TURN_OVERRIDE_SEC", "1.1"))     # giữ lệnh rẽ bao lâu
CAM_AVOID_COOLDOWN   = float(os.environ.get("CAM_AVOID_COOLDOWN", "1.0"))    # chống trigger liên tục

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
# Motion wrapper + LED (robust)
# =========================
class RobotMotion:
    """
    Supports STOP / FORWARD / BACK / TURN_LEFT / TURN_RIGHT
    LED: try multiple API variants to avoid "no light" issue.
    """
    def __init__(self, motion: MotionController, state: RobotState):
        self.motion = motion
        self.state = state
        self.dog = getattr(motion, "dog", None)

    def _try_led_methods(self, color: str, bps: float = 0.5):
        if not self.dog:
            return

        # 1) rgb_strip.set_mode("breath", color, bps=?)
        try:
            rs = getattr(self.dog, "rgb_strip", None)
            if rs:
                try:
                    rs.set_mode("breath", color, bps=bps)
                    return
                except TypeError:
                    rs.set_mode("breath", color, bps)
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # 2) rgb_strip.set_color / fill / show
        try:
            rs = getattr(self.dog, "rgb_strip", None)
            if rs:
                for fn in ("set_color", "fill"):
                    if hasattr(rs, fn):
                        try:
                            getattr(rs, fn)(color)
                            if hasattr(rs, "show"):
                                rs.show()
                            return
                        except Exception:
                            pass
        except Exception:
            pass

        # 3) rgb_led.set_color / set_rgb
        try:
            rl = getattr(self.dog, "rgb_led", None)
            if rl:
                for fn in ("set_color", "set_rgb", "setColor"):
                    if hasattr(rl, fn):
                        try:
                            getattr(rl, fn)(color)
                            return
                        except Exception:
                            pass
        except Exception:
            pass

    def set_led(self, color: str, bps: float = 0.5):
        try:
            self._try_led_methods(color=color, bps=bps)
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
        # thử nhiều label vì tuỳ MotionController bạn dùng
        self.set_led("purple", bps=0.6)
        self.state.set("MOVE", "BACK")
        for cmd in ("BACK", "BACKWARD", "REVERSE"):
            try:
                self.motion.execute(cmd)
                return
            except Exception:
                continue

    def turn_left(self):
        self.set_led("yellow", bps=0.6)
        self.state.set("TURN", "TURN_LEFT")
        try:
            self.motion.execute("TURN_LEFT")
        except Exception:
            pass

    def turn_right(self):
        self.set_led("yellow", bps=0.6)
        self.state.set("TURN", "TURN_RIGHT")
        try:
            self.motion.execute("TURN_RIGHT")
        except Exception:
            pass

# =========================
# LiDAR parsing helpers
# =========================
def _to_cm(dist_any: Any) -> Optional[float]:
    try:
        d = float(dist_any)
    except Exception:
        return None
    if d <= 0:
        return None
    # heuristics: m -> cm
    if d < 20.0:
        return d * 100.0
    # mm-ish -> cm
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
# LiDAR status / decision
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
    if t in ("TURNLEFT", "TURN_LEFT"):
        return "TURN_LEFT"
    if t in ("TURNRIGHT", "TURN_RIGHT"):
        return "TURN_RIGHT"
    if t in ("STOP",):
        return "STOP"
    return "STOP"

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
# Camera decision
# =========================
def camera_decision() -> Dict[str, Any]:
    """
    Expect payload from /take_camera_decision (VisionObstacleFusion):
    {
      ok: bool,
      ts: ...,
      decision: { has_obstacle, zone, obstacle_type, confidence, source, ... },
      boxes: [...]
    }
    """
    js = http_get_json(URL_CAMERA_DECISION, timeout=0.35)
    if not isinstance(js, dict) or not js.get("ok", False):
        return {"ok": False, "has_obstacle": False, "zone": "NONE", "confidence": 0.0, "obstacle_type": "none"}

    dec = js.get("decision", {})
    if not isinstance(dec, dict):
        dec = {}

    has_obs = bool(dec.get("has_obstacle", False))
    zone = str(dec.get("zone", "NONE") or "NONE").upper()
    conf = 0.0
    try:
        conf = float(dec.get("confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0

    return {
        "ok": True,
        "has_obstacle": has_obs,
        "zone": zone,
        "confidence": conf,
        "obstacle_type": str(dec.get("obstacle_type", "none")),
        "source": str(dec.get("source", "camera")),
        "raw": js,
    }

def pick_turn_direction(clearance: Dict[str, float], cam_zone: str) -> str:
    """
    Chọn hướng ít vật cản hơn:
    - dựa trên LiDAR LEFT/RIGHT clearance
    - tránh phía cam_zone báo có vật cản (penalty)
    """
    L = float(clearance.get("LEFT", 9999.0) or 9999.0)
    R = float(clearance.get("RIGHT", 9999.0) or 9999.0)

    scoreL = L
    scoreR = R

    z = (cam_zone or "NONE").upper()
    if z == "LEFT":
        scoreL -= 120.0
    elif z == "RIGHT":
        scoreR -= 120.0
    elif z == "CENTER":
        scoreL -= 60.0
        scoreR -= 60.0

    # nếu một bên cực gần thì phạt thêm
    if L < 60: scoreL -= 80
    if R < 60: scoreR -= 80

    return "TURN_LEFT" if scoreL >= scoreR else "TURN_RIGHT"

# =========================
# Map Web (debug only)
# =========================
class MapServer:
    def __init__(self):
        self.app = Flask("demo_map_2d_simple")
        self.lock = threading.Lock()

        self.ts = time.time()
        self.last_points: List[Tuple[float, float]] = []
        self.last_clearance: Dict[str, float] = {}
        self.last_cmd: str = "STOP"
        self.robot_state: str = "BOOT"
        self.last_lidar_dec_raw: str = ""
        self.last_lidar_dec_norm: str = ""
        self.last_reason: str = ""
        self.last_pts_n: int = 0

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
                        "reason": self.last_reason,
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
    <b>2D Lidar Map (simple)</b> — refresh 5 fps — URL: http://&lt;pi_ip&gt;:{MAP_PORT}/
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

      <div class="kv"><span class="k">pts_n:</span> <span id="pn">-</span></div>
      <div class="kv"><span class="k">lidar_dec_raw:</span> <span id="ldr">-</span></div>
      <div class="kv"><span class="k">lidar_dec_norm:</span> <span id="ldn">-</span></div>
      <div class="kv"><span class="k">reason:</span> <span id="rs">-</span></div>
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
        reason: str = "",
    ):
        with self.lock:
            self.last_points = points
            self.last_pts_n = len(points)
            self.last_clearance = clearance
            self.last_cmd = cmd
            self.robot_state = robot_state
            self.last_lidar_dec_raw = lidar_dec_raw or ""
            self.last_lidar_dec_norm = lidar_dec_norm or ""
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

    # avoidance state
    turn_override_until = 0.0
    turn_override_cmd = "STOP"
    last_cam_avoid_ts = 0.0

    print(f"[DEMO] map web: http://<pi_ip>:{MAP_PORT}/", flush=True)
    print("[DEMO] control mode: LiDAR decision + camera override (/take_camera_decision)", flush=True)
    print(f"[DEMO] camera endpoint: {URL_CAMERA_DECISION}", flush=True)

    try:
        while True:
            now = time.time()

            pts, _scan_js = lidar_scan_points()
            clr = lidar_clearance(pts)

            lidar_dec_raw, lidar_dec_norm = lidar_decision_label_text()

            # If LiDAR not reachable -> STOP
            if not lidar_dec_norm:
                base_cmd = "STOP"
                reason = "no_lidar_decision -> STOP"
            else:
                base_cmd = lidar_dec_norm
                reason = "lidar_decision_label"

            # camera decision (optional)
            cam = camera_decision()
            cam_has = bool(cam.get("has_obstacle", False))
            cam_zone = str(cam.get("zone", "NONE"))
            cam_conf = float(cam.get("confidence", 0.0) or 0.0)

            front_cm = float(clr.get("FRONT", 9999.0) or 9999.0)
            lidar_clear = (base_cmd == "FORWARD") and (front_cm >= LIDAR_FRONT_CLEAR_CM)

            # 1) if we are in override window -> keep turning
            if now < turn_override_until:
                last_cmd = turn_override_cmd
                reason = f"cam_override_window cmd={turn_override_cmd}"

            else:
                # 2) if lidar says clear but camera sees obstacle -> STOP + BACK + TURN
                should_cam_avoid = (
                    lidar_clear
                    and cam.get("ok", False)
                    and cam_has
                    and (cam_conf >= CAM_OBS_MIN_CONF)
                    and ((now - last_cam_avoid_ts) >= CAM_AVOID_COOLDOWN)
                )

                if should_cam_avoid:
                    last_cam_avoid_ts = now

                    # STOP immediately (force)
                    rm.stop()
                    last_sent_cmd, last_sent_ts = "STOP", time.time()

                    # BACK for BACK_SEC (force)
                    rm.back()
                    last_sent_cmd, last_sent_ts = "BACK", time.time()
                    time.sleep(max(0.1, BACK_SEC))

                    rm.stop()
                    last_sent_cmd, last_sent_ts = "STOP", time.time()

                    # choose turn direction (least obstacles)
                    turn_cmd = pick_turn_direction(clr, cam_zone)

                    # set override window
                    turn_override_cmd = turn_cmd
                    turn_override_until = time.time() + TURN_OVERRIDE_SEC

                    last_cmd = turn_cmd
                    reason = f"cam_detected({cam_zone},conf={cam_conf:.2f}) -> STOP+BACK -> {turn_cmd}"

                    # optional face feedback
                    try:
                        set_face("angry")
                    except Exception:
                        pass

                else:
                    # 3) normal follow lidar
                    last_cmd = base_cmd
                    # attach camera info into reason (debug)
                    if cam.get("ok", False):
                        reason = f"{reason} | cam={('OBS' if cam_has else 'clear')} {cam_zone} c={cam_conf:.2f}"

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
                reason=reason,
            )

            # Debug log
            if (now - last_dbg_ts) >= DBG_PRINT_EVERY_SEC:
                last_dbg_ts = now
                print(
                    f"[DBG] front={front_cm:.1f} L={clr.get('LEFT',9999.0):.1f} R={clr.get('RIGHT',9999.0):.1f} "
                    f"lidar_raw={lidar_dec_raw} -> base={base_cmd} | cmd={last_cmd} | cam_ok={cam.get('ok',False)} "
                    f"cam_obs={cam_has} zone={cam_zone} conf={cam_conf:.2f} | state={state} | reason={reason}",
                    flush=True
                )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)

    finally:
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
