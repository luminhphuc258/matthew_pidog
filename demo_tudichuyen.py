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

# Lidar server (localhost:9399)
LIDAR_BASE = "http://127.0.0.1:9399"
URL_LIDAR_STATUS   = f"{LIDAR_BASE}/api/status"
URL_LIDAR_DECISION = f"{LIDAR_BASE}/api/decision_label"
URL_LIDAR_DATA     = f"{LIDAR_BASE}/take_lidar_data"

# Gesture + camera decision (localhost:8000)
GEST_BASE = "http://127.0.0.1:8000"
URL_GESTURE         = f"{GEST_BASE}/take_gesture_meaning"
URL_CAMERA_DECISION = f"{GEST_BASE}/take_camera_decision"

# Map web
MAP_PORT = 5000

# Control loop
LOOP_HZ = 20.0

# --- HTTP timeouts ---
HTTP_TIMEOUT_SCAN = 1.0
HTTP_TIMEOUT_DEC  = 0.7
HTTP_TIMEOUT_CAM  = 0.9
HTTP_TIMEOUT_GEST = 0.4

# --- Poll rates (IMPORTANT) ---
# giảm rate gọi endpoint để tránh timeout/rỗng
SCAN_PERIOD_SEC     = 0.15   # ~6.6Hz
DECISION_PERIOD_SEC = 0.20   # 5Hz
CAM_PERIOD_SEC      = 0.30   # ~3.3Hz (chỉ gọi khi chuẩn bị đi thẳng)

# Command refresh (reduce jitter)
CMD_REFRESH_SEC = 0.35  # resend same cmd at most every 0.35s

# Safety distances (cm)
SAFE_FORWARD_CM = 65.0
SAFE_TURN_CM = 45.0
EMERGENCY_STOP_CM = 18.0

# Gesture TTL
GESTURE_TTL_SEC = 1.2

# Debug logging
DEBUG_LOG = True
DEBUG_LOG_PERIOD_SEC = 1.0   # in log 1 lần/giây

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
_session = requests.Session()

def http_get_json(url: str, timeout: float) -> Optional[dict]:
    try:
        r = _session.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# =========================
# Robot Status
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
# Motion wrapper (NO pose apply, NO sit)
# =========================
class RobotMotion:
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
        self.motion.boot()  # MotionController.boot đã load pose sẵn
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
        self.set_led("red", bps=0.55)
        self.state.set("BACK", "BACK")
        try:
            self.motion.execute("BACK")
        except Exception:
            try:
                self.motion.execute("BACKWARD")
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
            js = http_get_json(URL_GESTURE, timeout=HTTP_TIMEOUT_GEST)
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
# Lidar parsing helpers (MAKE IT MORE TOLERANT)
# =========================
def _to_cm(dist: float) -> float:
    """
    Heuristic convert distance to cm.
    - if value looks like mm (e.g. 1200..8000), convert to cm
    - if value looks like meters (0.2..12), convert to cm
    - else assume already cm
    """
    d = float(dist)
    if d <= 0:
        return d

    # meters
    if 0.02 < d < 30.0:
        return d * 100.0

    # mm
    if d >= 200.0:
        # could be mm or cm; if huge like 3000 => mm
        if d > 500.0:
            return d / 10.0
    return d

def _extract_points(scan_json: Any) -> List[Tuple[float, float]]:
    """
    Return list of (angle_deg, dist_cm) from many possible formats.
    """
    if not isinstance(scan_json, dict):
        return []

    # common containers
    for key in ("points", "scan", "data", "lidar", "result"):
        arr = scan_json.get(key, None)
        if isinstance(arr, list) and arr:
            pts: List[Tuple[float, float]] = []
            for it in arr:
                # case dict
                if isinstance(it, dict):
                    a = it.get("angle", it.get("deg", it.get("theta", it.get("a", None))))
                    d = it.get("dist_cm",
                               it.get("distance_cm",
                               it.get("dist",
                               it.get("distance",
                               it.get("r",
                               it.get("dist_mm",
                               it.get("distance_mm", None)))))))
                    if a is None or d is None:
                        continue
                    try:
                        pts.append((float(a), _to_cm(float(d))))
                    except Exception:
                        pass
                # case [angle, dist]
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    try:
                        a = float(it[0])
                        d = _to_cm(float(it[1]))
                        pts.append((a, d))
                    except Exception:
                        pass

            if pts:
                return pts

    # arrays style: angles + dists
    ang = scan_json.get("angles", scan_json.get("angle_list", None))
    dist = scan_json.get("dists_cm",
                         scan_json.get("distances_cm",
                         scan_json.get("dists",
                         scan_json.get("distances",
                         scan_json.get("dists_mm",
                         scan_json.get("distances_mm", None))))))

    if isinstance(ang, list) and isinstance(dist, list) and len(ang) == len(dist) and len(ang) > 0:
        pts: List[Tuple[float, float]] = []
        for a, d in zip(ang, dist):
            try:
                pts.append((float(a), _to_cm(float(d))))
            except Exception:
                pass
        return pts

    return []

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

def pick_direction_by_clearance(clr: Dict[str, float], prefer: str = "FRONT") -> str:
    front = clr.get("FRONT", 0.0)
    left  = clr.get("LEFT", 0.0)
    right = clr.get("RIGHT", 0.0)
    back  = clr.get("BACK", 0.0)

    if min(front, left, right) < EMERGENCY_STOP_CM:
        if back > SAFE_TURN_CM:
            return "BACK"
        return "STOP"

    if prefer == "FRONT" and front >= SAFE_FORWARD_CM:
        return "FORWARD"

    candidates = [("FORWARD", front), ("TURN_LEFT", left), ("TURN_RIGHT", right), ("BACK", back)]
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_cmd, best_val = candidates[0]

    if best_cmd == "FORWARD" and best_val >= SAFE_FORWARD_CM:
        return "FORWARD"
    if best_cmd in ("TURN_LEFT", "TURN_RIGHT") and best_val >= SAFE_TURN_CM:
        return best_cmd
    if best_cmd == "BACK" and best_val >= SAFE_TURN_CM:
        return "BACK"

    return "STOP"

# =========================
# Sensor cache (avoid over-polling)
# =========================
class SensorCache:
    def __init__(self):
        self.lock = threading.Lock()

        self.last_scan_pts: List[Tuple[float, float]] = []
        self.last_scan_raw: Optional[dict] = None
        self.last_scan_ts: float = 0.0

        self.last_lidar_dec: str = "FORWARD"
        self.last_lidar_dec_raw: Optional[dict] = None
        self.last_lidar_dec_ts: float = 0.0

        self.last_cam_clear: bool = True
        self.last_cam_raw: Optional[dict] = None
        self.last_cam_ts: float = 0.0

    def update_scan(self):
        js = http_get_json(URL_LIDAR_DATA, timeout=HTTP_TIMEOUT_SCAN)
        now = time.time()
        if js:
            pts = _extract_points(js)
        else:
            pts = []
        with self.lock:
            # keep last good points if this poll fails (to avoid 9999 forever)
            if pts:
                self.last_scan_pts = pts
                self.last_scan_raw = js
                self.last_scan_ts = now
            else:
                # still store raw for debug
                self.last_scan_raw = js
                # do NOT overwrite last_scan_pts when empty (helps stability)

    def update_lidar_decision(self):
        js = http_get_json(URL_LIDAR_DECISION, timeout=HTTP_TIMEOUT_DEC)
        now = time.time()
        dec = None
        if js:
            for k in ("decision", "label", "decision_label", "move"):
                v = js.get(k, None)
                if v:
                    dec = str(v).upper().strip()
                    break
        if not dec:
            dec = "FORWARD"
        if dec == "BACKWARD":
            dec = "BACK"

        with self.lock:
            self.last_lidar_dec = dec
            self.last_lidar_dec_raw = js
            self.last_lidar_dec_ts = now

    def update_camera(self):
        js = http_get_json(URL_CAMERA_DECISION, timeout=HTTP_TIMEOUT_CAM)
        now = time.time()

        # parse tolerant
        clear = True  # fail-open
        label = ""
        if js and js.get("ok") is not None:
            # try various keys
            label = str(js.get("label", js.get("decision", js.get("result", ""))) or "").lower().strip()

            # normalize
            if label in ("no obstacle", "clear", "free", "ok", "none"):
                clear = True
            elif label in ("obstacle", "have obstacle", "yes have obstacle", "blocked", "stop"):
                clear = False
            else:
                # some services return boolean
                if isinstance(js.get("clear", None), bool):
                    clear = bool(js["clear"])
                elif isinstance(js.get("obstacle", None), bool):
                    clear = (not bool(js["obstacle"]))
                else:
                    clear = True

        with self.lock:
            self.last_cam_clear = clear
            self.last_cam_raw = js
            self.last_cam_ts = now

    def get(self):
        with self.lock:
            return (
                list(self.last_scan_pts),
                self.last_scan_raw,
                self.last_scan_ts,
                self.last_lidar_dec,
                self.last_lidar_dec_raw,
                self.last_lidar_dec_ts,
                self.last_cam_clear,
                self.last_cam_raw,
                self.last_cam_ts
            )

# =========================
# Lidar ready
# =========================
def lidar_ready() -> bool:
    js = http_get_json(URL_LIDAR_STATUS, timeout=0.8)
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

# =========================
# Map Web (2D TOP-DOWN)
# =========================
class MapServer:
    def __init__(self):
        self.app = Flask("demo_map")
        self.lock = threading.Lock()
        self.last_points: List[Tuple[float, float]] = []
        self.last_clearance: Dict[str, float] = {}
        self.last_cmd: str = "STOP"
        self.robot_state: str = "BOOT"
        self.ts = time.time()
        self.debug_line: str = ""

        @self.app.get("/map.json")
        def map_json():
            with self.lock:
                payload = {
                    "ts": self.ts,
                    "points": [{"angle": a, "dist_cm": d} for a, d in self.last_points[:2500]],
                    "clearance": self.last_clearance,
                    "cmd": self.last_cmd,
                    "robot_state": self.robot_state,
                    "debug": self.debug_line,
                }
            return jsonify(payload)

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

    def _html(self) -> str:
        # 2D map: robot bottom-center, forward up
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Robot Lidar 2D Map</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }}
    .bar {{ padding:10px 14px; background:#111827; position:sticky; top:0; border-bottom:1px solid #223; }}
    .wrap {{ display:flex; gap:14px; padding:14px; }}
    canvas {{ background:#05070a; border:1px solid #223; border-radius:10px; }}
    .card {{ background:#0f172a; border:1px solid #223; border-radius:10px; padding:12px; min-width:300px; }}
    .kv {{ margin:6px 0; }}
    .k {{ color:#93c5fd; }}
    .small {{ font-size:12px; color:#aab; white-space:pre-wrap; }}
  </style>
</head>
<body>
  <div class="bar">
    <b>2D Lidar Map</b> — refresh 5 fps — URL: http://&lt;pi_ip&gt;:{MAP_PORT}/
  </div>

  <div class="wrap">
    <canvas id="cv" width="920" height="560"></canvas>
    <div class="card">
      <div class="kv"><span class="k">Robot State:</span> <span id="st">-</span></div>
      <div class="kv"><span class="k">Cmd:</span> <span id="cmd">-</span></div>
      <div class="kv"><span class="k">Pts:</span> <span id="n">-</span></div>
      <div class="kv"><span class="k">FRONT:</span> <span id="f">-</span> cm</div>
      <div class="kv"><span class="k">LEFT:</span> <span id="l">-</span> cm</div>
      <div class="kv"><span class="k">RIGHT:</span> <span id="r">-</span> cm</div>
      <div class="kv"><span class="k">BACK:</span> <span id="b">-</span> cm</div>
      <div class="kv"><span class="k">TS:</span> <span id="ts">-</span></div>
      <hr style="border:0;border-top:1px solid #223;margin:10px 0;">
      <div class="small" id="dbg"></div>
    </div>
  </div>

<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');

function draw(data) {{
  ctx.clearRect(0,0,cv.width,cv.height);

  const ox = cv.width * 0.5;
  const oy = cv.height * 0.88;

  // scale: cm -> px
  const CM_TO_PX = 1.4;  // adjust zoom
  const MAX_CM = 300;    // draw up to 3m

  // grid circles every 50cm
  ctx.strokeStyle = '#142033';
  ctx.lineWidth = 1;
  for (let cm=50; cm<=MAX_CM; cm+=50) {{
    ctx.beginPath();
    ctx.arc(ox, oy, cm*CM_TO_PX, 0, Math.PI*2);
    ctx.stroke();
    ctx.fillStyle = '#0f172a';
    ctx.fillText(cm+"cm", ox+6, oy - cm*CM_TO_PX - 6);
  }}

  // forward axis
  ctx.strokeStyle = '#1f2a44';
  ctx.beginPath();
  ctx.moveTo(ox, oy);
  ctx.lineTo(ox, oy - MAX_CM*CM_TO_PX);
  ctx.stroke();

  // robot
  ctx.fillStyle = '#60a5fa';
  ctx.beginPath(); ctx.arc(ox, oy, 7, 0, Math.PI*2); ctx.fill();

  // points
  const pts = data.points || [];
  for (const p of pts) {{
    const ang = (p.angle || 0) * Math.PI/180.0;
    const d = (p.dist_cm || 0);
    if (!d || d <= 0) continue;
    if (d > MAX_CM) continue;

    // polar -> x,y (cm) : forward = +y
    const x_cm = Math.sin(ang) * d;
    const y_cm = Math.cos(ang) * d;

    const sx = ox + x_cm * CM_TO_PX;
    const sy = oy - y_cm * CM_TO_PX;

    // brightness: near brighter
    const b = Math.max(40, Math.min(255, 300 - d));
    ctx.fillStyle = `rgb(${{b}},${{b}},${{b}})`;
    ctx.fillRect(sx, sy, 2, 2);
  }}

  // UI
  document.getElementById('st').textContent = data.robot_state || '-';
  document.getElementById('cmd').textContent = data.cmd || '-';
  document.getElementById('ts').textContent = String(data.ts || '-');
  document.getElementById('n').textContent = String(pts.length || 0);

  const c = data.clearance || {{}};
  document.getElementById('f').textContent = (c.FRONT ?? '-');
  document.getElementById('l').textContent = (c.LEFT ?? '-');
  document.getElementById('r').textContent = (c.RIGHT ?? '-');
  document.getElementById('b').textContent = (c.BACK ?? '-');

  document.getElementById('dbg').textContent = data.debug || '';
}}

async function tick() {{
  try {{
    const r = await fetch('/map.json', {{cache:'no-store'}});
    const data = await r.json();
    draw(data);
  }} catch(e) {{}}
}}

setInterval(tick, 200);
tick();
</script>
</body>
</html>
"""

    def update(self, points: List[Tuple[float, float]], clearance: Dict[str, float], cmd: str, robot_state: str, debug_line: str):
        with self.lock:
            self.last_points = points
            self.last_clearance = clearance
            self.last_cmd = cmd
            self.robot_state = robot_state
            self.debug_line = debug_line
            self.ts = time.time()

    def run_bg(self, port: int):
        def _serve():
            try:
                self.app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
            except Exception as e:
                print(f"[MAP] ERROR: failed to start Flask on port {port}: {e}", flush=True)

        threading.Thread(target=_serve, daemon=True).start()
        print(f"[MAP] started on 0.0.0.0:{port}", flush=True)

# =========================
# Cmd sender (reduce jitter)
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

    gp = GesturePoller()
    gp.start()

    cache = SensorCache()

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

    # timers for sensor polling
    next_scan_ts = 0.0
    next_dec_ts  = 0.0
    next_log_ts  = 0.0
    next_cam_ts  = 0.0

    print(f"[DEMO] map web: http://<pi_ip>:{MAP_PORT}/", flush=True)

    try:
        while True:
            now = time.time()

            # --- poll sensors at lower rate ---
            if now >= next_scan_ts:
                cache.update_scan()
                next_scan_ts = now + SCAN_PERIOD_SEC

            if now >= next_dec_ts:
                cache.update_lidar_decision()
                next_dec_ts = now + DECISION_PERIOD_SEC

            pts, scan_raw, scan_ts, lidar_dec, dec_raw, dec_ts, cam_clear, cam_raw, cam_ts = cache.get()

            # compute clearance from last-known-good points
            clr = lidar_clearance(pts)

            # ===== 1) Gesture priority =====
            g = gp.get_active()
            if g:
                g = g.upper().strip()
                if g in ("STOP", "STOPMUSIC", "SIT"):
                    last_cmd = "STOP"
                elif g in ("MOVELEFT", "TURNLEFT"):
                    last_cmd = "TURN_LEFT"
                elif g in ("MOVERIGHT", "TURNRIGHT"):
                    last_cmd = "TURN_RIGHT"
                elif g in ("BACK", "BACKWARD"):
                    last_cmd = "BACK"
                else:
                    last_cmd = "STOP"
            else:
                # ===== 2) Auto by lidar + camera =====
                # emergency stop
                if clr.get("FRONT", 9999.0) < EMERGENCY_STOP_CM:
                    last_cmd = "STOP"
                else:
                    prefer_forward = ("FORWARD" in lidar_dec or lidar_dec == "FORWARD")

                    # only call camera when we are about to go forward and it's safe by lidar
                    if prefer_forward and clr.get("FRONT", 0.0) >= SAFE_FORWARD_CM:
                        if now >= next_cam_ts:
                            cache.update_camera()
                            next_cam_ts = now + CAM_PERIOD_SEC
                            pts, scan_raw, scan_ts, lidar_dec, dec_raw, dec_ts, cam_clear, cam_raw, cam_ts = cache.get()

                        if cam_clear:
                            last_cmd = "FORWARD"
                        else:
                            last_cmd = pick_direction_by_clearance(clr, prefer="LEFT")
                    else:
                        if lidar_dec in ("TURN_LEFT", "TURN_RIGHT", "BACK", "STOP"):
                            last_cmd = lidar_dec
                        else:
                            last_cmd = pick_direction_by_clearance(clr, prefer="FRONT")

            # send movement rate-limited
            last_sent_cmd, last_sent_ts = send_cmd_rate_limited(
                rm=rm, cmd=last_cmd, last_sent_cmd=last_sent_cmd, last_sent_ts=last_sent_ts
            )

            # ----- DEBUG LOG (throttle) -----
            if DEBUG_LOG and (now >= next_log_ts):
                npts = len(pts)
                front = clr.get("FRONT", 9999.0)
                left  = clr.get("LEFT", 9999.0)
                right = clr.get("RIGHT", 9999.0)

                # keys snapshot
                scan_keys = list(scan_raw.keys())[:10] if isinstance(scan_raw, dict) else []
                dec_keys  = list(dec_raw.keys())[:10]  if isinstance(dec_raw, dict) else []
                cam_keys  = list(cam_raw.keys())[:10]  if isinstance(cam_raw, dict) else []

                print(
                    f"[DBG] pts={npts} front={front:.1f} left={left:.1f} right={right:.1f} "
                    f"lidar_dec={lidar_dec} cam_clear={cam_clear} -> cmd={last_cmd} "
                    f"| scan_keys={scan_keys} dec_keys={dec_keys} cam_keys={cam_keys}",
                    flush=True
                )
                next_log_ts = now + DEBUG_LOG_PERIOD_SEC

            # show a compact debug line on the web panel
            dbg = (
                f"pts={len(pts)} | lidar_dec={lidar_dec} | cam_clear={cam_clear}\n"
                f"scan_age={max(0.0, now-scan_ts):.2f}s | dec_age={max(0.0, now-dec_ts):.2f}s | cam_age={max(0.0, now-cam_ts):.2f}s\n"
                f"FRONT={clr.get('FRONT', 0):.1f} LEFT={clr.get('LEFT', 0):.1f} RIGHT={clr.get('RIGHT', 0):.1f}"
            )

            map_server.update(points=pts, clearance=clr, cmd=last_cmd, robot_state=str(state), debug_line=dbg)

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
