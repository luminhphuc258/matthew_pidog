#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import threading
import subprocess
import socket
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
from flask import Flask, jsonify, Response

from motion_controller import MotionController

# =========================
# CONFIG
# =========================
POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# Lidar server (localhost:9399)
LIDAR_BASE = "http://127.0.0.1:9399"
URL_LIDAR_STATUS = f"{LIDAR_BASE}/api/status"
URL_LIDAR_DECISION = f"{LIDAR_BASE}/api/decision_label"
URL_LIDAR_DATA = f"{LIDAR_BASE}/take_lidar_data"

# Gesture + camera decision (localhost:8000)
GEST_BASE = "http://127.0.0.1:8000"
URL_GESTURE = f"{GEST_BASE}/take_gesture_meaning"
URL_CAMERA_DECISION = f"{GEST_BASE}/take_camera_decision"

# Map web
MAP_PORT = 5000

# Loop
LOOP_HZ = 20.0
HTTP_TIMEOUT = 0.6

# Safety distances (cm)
SAFE_FORWARD_CM = 65.0
SAFE_TURN_CM = 45.0
EMERGENCY_STOP_CM = 18.0

# =========================
# REST SCHEDULE (as you requested)
# =========================
# Random time until next rest: 8..15 minutes
REST_INTERVAL_MIN_MINUTES = 8
REST_INTERVAL_MAX_MINUTES = 15

# At rest time: SIT and hold random 9..15 minutes
REST_HOLD_MIN = 9
REST_HOLD_MAX = 15
REST_HOLD_UNIT = "min"  # keep "min" as your latest requirement
# If you want 9..15 seconds instead, set:
# REST_HOLD_MIN = 9
# REST_HOLD_MAX = 15
# REST_HOLD_UNIT = "sec"

# Gesture TTL
GESTURE_TTL_SEC = 1.2

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
# Utils
# =========================
def run(cmd: List[str]):
    return subprocess.run(cmd, check=False)

def set_volumes():
    run(["amixer", "-q", "sset", "robot-hat speaker", "100%"])
    run(["amixer", "-q", "sset", "robot-hat speaker Playback Volume", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic Capture Volume", "100%"])

def http_get_json(url: str, timeout: float = HTTP_TIMEOUT) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def seconds_from_hold_value(v: float) -> float:
    if REST_HOLD_UNIT == "sec":
        return float(v)
    # default minutes
    return float(v) * 60.0

# =========================
# Robot Status Manager
# =========================
class RobotState:
    """
    Lưu trạng thái để theo dõi SIT/...
    """
    def __init__(self):
        self.mode = "BOOT"   # BOOT / STAND / SIT / MOVE / TURN / BACK / STOP
        self.detail = ""
        self.ts = time.time()

    def set(self, mode: str, detail: str = ""):
        self.mode = mode
        self.detail = detail
        self.ts = time.time()

    def is_sitting(self) -> bool:
        return self.mode == "SIT"

    def __repr__(self):
        return f"{self.mode}:{self.detail}" if self.detail else self.mode

# =========================
# Motion wrapper
# =========================
class RobotMotion:
    """
    IMPORTANT RULES (per your request):
    - boot() does NOT apply pose config outside (MotionController.boot already did)
    - ONLY do support_stand when transitioning SIT -> STAND
    - NO pose/stabilize for FORWARD/TURN/BACK
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
        """
        boot() only. NO apply pose here.
        """
        self.state.set("BOOT")
        self.motion.boot()
        set_volumes()
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

    def support_stand_from_sit(self):
        """
        ONLY used when SIT -> STAND.
        """
        t0 = time.time()
        while time.time() - t0 < 0.8:
            try:
                self.motion.execute("STANDUP")
            except Exception:
                try:
                    self.motion.execute("STOP")
                except Exception:
                    pass
            time.sleep(0.02)

    def sit_down_and_hold(self, hold_seconds: float):
        """
        Stop -> SIT -> hold (no stabilize except when standing up again)
        """
        try:
            self.motion.execute("STOP")
        except Exception:
            pass
        time.sleep(0.1)

        # SIT
        t0 = time.time()
        while time.time() - t0 < 0.9:
            try:
                self.motion.execute("SIT")
            except Exception:
                pass
            time.sleep(0.02)

        self.state.set("SIT")
        self.set_led("red", bps=0.35)
        set_face("sleep")

        # hold
        t_end = time.time() + float(hold_seconds)
        while time.time() < t_end:
            time.sleep(0.2)

    def stand_from_sit_only(self):
        """
        EXACT requirement:
        - ONLY stabilize/support stand when SIT -> STAND
        """
        if not self.state.is_sitting():
            return
        set_face("what_is_it")
        self.support_stand_from_sit()
        try:
            self.motion.execute("STOP")
        except Exception:
            pass
        self.set_led("white", bps=0.35)
        self.state.set("STAND")

    # Movement commands (NO stabilize inside)
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
def _extract_points(scan_json: dict) -> List[Tuple[float, float]]:
    if not isinstance(scan_json, dict):
        return []
    pts: List[Tuple[float, float]] = []
    for key in ("points", "scan", "data"):
        arr = scan_json.get(key, None)
        if isinstance(arr, list) and arr:
            for it in arr:
                if isinstance(it, dict):
                    a = it.get("angle", it.get("deg", it.get("theta", None)))
                    d = it.get("dist_cm", it.get("distance_cm", it.get("dist", it.get("r", None))))
                    if a is None or d is None:
                        continue
                    try:
                        pts.append((float(a), float(d)))
                    except Exception:
                        pass
            if pts:
                return pts

    ang = scan_json.get("angles", None)
    dist = scan_json.get("dists_cm", scan_json.get("distances_cm", scan_json.get("dists", None)))
    if isinstance(ang, list) and isinstance(dist, list) and len(ang) == len(dist):
        for a, d in zip(ang, dist):
            try:
                pts.append((float(a), float(d)))
            except Exception:
                pass
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

def pick_direction_by_clearance(clr: Dict[str, float], prefer: str = "FRONT") -> str:
    front = clr.get("FRONT", 0.0)
    left = clr.get("LEFT", 0.0)
    right = clr.get("RIGHT", 0.0)
    back = clr.get("BACK", 0.0)

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
# Camera Decision
# =========================
def camera_is_clear() -> bool:
    js = http_get_json(URL_CAMERA_DECISION, timeout=0.45)
    if not js or not js.get("ok"):
        return True
    label = str(js.get("label", "") or "").lower().strip()
    return (label == "no obstacle")

# =========================
# Lidar status/decision
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

def lidar_decision_label() -> Optional[str]:
    js = http_get_json(URL_LIDAR_DECISION, timeout=0.55)
    if not js:
        return None
    for k in ("decision", "label", "decision_label", "move"):
        v = js.get(k, None)
        if v:
            return str(v).upper().strip()
    return None

def lidar_scan_points() -> List[Tuple[float, float]]:
    js = http_get_json(URL_LIDAR_DATA, timeout=0.65)
    if not js:
        return []
    return _extract_points(js)

# =========================
# Map Web (fake 3D)
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

        @self.app.get("/map.json")
        def map_json():
            with self.lock:
                payload = {
                    "ts": self.ts,
                    "points": [{"angle": a, "dist_cm": d} for a, d in self.last_points[:2000]],
                    "clearance": self.last_clearance,
                    "cmd": self.last_cmd,
                    "robot_state": self.robot_state,
                }
            return jsonify(payload)

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

    def _html(self) -> str:
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Robot Lidar Fake 3D Map</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }}
    .bar {{ padding:10px 14px; background:#111827; position:sticky; top:0; border-bottom:1px solid #223; }}
    .wrap {{ display:flex; gap:14px; padding:14px; }}
    canvas {{ background:#05070a; border:1px solid #223; border-radius:10px; }}
    .card {{ background:#0f172a; border:1px solid #223; border-radius:10px; padding:12px; min-width:260px; }}
    .kv {{ margin:6px 0; }}
    .k {{ color:#93c5fd; }}
  </style>
</head>
<body>
  <div class="bar">
    <b>Fake 3D Map</b> — refresh 5 fps — URL: http://&lt;pi_ip&gt;:{MAP_PORT}/
  </div>

  <div class="wrap">
    <canvas id="cv" width="820" height="560"></canvas>
    <div class="card">
      <div class="kv"><span class="k">Robot State:</span> <span id="st">-</span></div>
      <div class="kv"><span class="k">Cmd:</span> <span id="cmd">-</span></div>
      <div class="kv"><span class="k">FRONT:</span> <span id="f">-</span> cm</div>
      <div class="kv"><span class="k">LEFT:</span> <span id="l">-</span> cm</div>
      <div class="kv"><span class="k">RIGHT:</span> <span id="r">-</span> cm</div>
      <div class="kv"><span class="k">BACK:</span> <span id="b">-</span> cm</div>
      <div class="kv"><span class="k">TS:</span> <span id="ts">-</span></div>
      <hr style="border:0;border-top:1px solid #223;margin:10px 0;">
      <div style="font-size:13px; color:#aab;">
        Tip: điểm càng sáng = càng gần robot.
      </div>
    </div>
  </div>

<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');

function draw(data) {{
  ctx.clearRect(0,0,cv.width,cv.height);

  const ox = cv.width * 0.5;
  const oy = cv.height * 0.78;

  ctx.strokeStyle = '#142033';
  ctx.lineWidth = 1;
  for (let i=1;i<=6;i++) {{
    const rr = i * 70;
    ctx.beginPath();
    ctx.ellipse(ox, oy, rr, rr*0.55, 0, 0, Math.PI*2);
    ctx.stroke();
  }}

  ctx.fillStyle = '#60a5fa';
  ctx.beginPath(); ctx.arc(ox, oy, 6, 0, Math.PI*2); ctx.fill();

  const pts = data.points || [];
  for (const p of pts) {{
    const ang = (p.angle || 0) * Math.PI/180.0;
    const d = (p.dist_cm || 0);
    if (!d || d <= 0) continue;

    const x = Math.sin(ang) * d;
    const y = Math.cos(ang) * d;

    const depth = Math.max(1, d);
    const scale = 1.0 / (1.0 + depth/260.0);

    const sx = ox + x * 1.3 * scale;
    const sy = oy - y * 1.1 * scale;

    const b = Math.max(40, Math.min(255, 300 - d));
    ctx.fillStyle = `rgb(${b},${b},${b})`;
    ctx.fillRect(sx, sy, 2, 2);
  }}

  document.getElementById('st').textContent = data.robot_state || '-';
  document.getElementById('cmd').textContent = data.cmd || '-';
  document.getElementById('ts').textContent = String(data.ts || '-');

  const c = data.clearance || {{}};
  document.getElementById('f').textContent = (c.FRONT ?? '-');
  document.getElementById('l').textContent = (c.LEFT ?? '-');
  document.getElementById('r').textContent = (c.RIGHT ?? '-');
  document.getElementById('b').textContent = (c.BACK ?? '-');
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

    def update(self, points: List[Tuple[float, float]], clearance: Dict[str, float], cmd: str, robot_state: str):
        with self.lock:
            self.last_points = points
            self.last_clearance = clearance
            self.last_cmd = cmd
            self.robot_state = robot_state
            self.ts = time.time()

    def run_bg(self):
        threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=MAP_PORT, debug=False, use_reloader=False),
            daemon=True
        ).start()

# =========================
# Main autopilot
# =========================
def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    state = RobotState()
    motion = MotionController(pose_file=POSE_FILE)
    rm = RobotMotion(motion=motion, state=state)

    map_server = MapServer()
    map_server.run_bg()

    gp = GesturePoller()
    gp.start()

    # BOOT only (NO apply pose outside)
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

    # next rest time: random 8..15 minutes
    next_rest_ts = time.time() + random.randint(REST_INTERVAL_MIN_MINUTES * 60, REST_INTERVAL_MAX_MINUTES * 60)

    dt = 1.0 / LOOP_HZ
    last_cmd = "STOP"

    print(f"[DEMO] map web: http://<pi_ip>:{MAP_PORT}/", flush=True)

    try:
        while True:
            now = time.time()

            # ===== periodic rest =====
            if now >= next_rest_ts:
                # sit hold random 9..15 minutes (or seconds if you change REST_HOLD_UNIT)
                hold_val = random.randint(REST_HOLD_MIN, REST_HOLD_MAX)
                hold_seconds = seconds_from_hold_value(hold_val)
                print(f"[REST] sit hold = {hold_val}{REST_HOLD_UNIT}", flush=True)

                rm.sit_down_and_hold(hold_seconds=hold_seconds)

                # IMPORTANT: only stabilize when SIT -> STAND
                rm.stand_from_sit_only()

                # schedule next rest: random 8..15 minutes
                next_rest_ts = time.time() + random.randint(REST_INTERVAL_MIN_MINUTES * 60, REST_INTERVAL_MAX_MINUTES * 60)

            # get lidar scan
            pts = lidar_scan_points()
            clr = lidar_clearance(pts)

            # update map (state + last_cmd)
            map_server.update(points=pts, clearance=clr, cmd=last_cmd, robot_state=str(state))

            # ===== 1) Gesture priority =====
            g = gp.get_active()
            if g:
                g = g.upper().strip()

                if g in ("STOP", "STOPMUSIC"):
                    rm.stop()
                    last_cmd = "STOP"

                elif g == "SIT":
                    # gesture SIT: hold random 9..15 minutes
                    hold_val = random.randint(REST_HOLD_MIN, REST_HOLD_MAX)
                    hold_seconds = seconds_from_hold_value(hold_val)
                    print(f"[GESTURE] SIT hold = {hold_val}{REST_HOLD_UNIT}", flush=True)

                    rm.sit_down_and_hold(hold_seconds=hold_seconds)
                    rm.stand_from_sit_only()
                    last_cmd = "STOP"

                elif g == "STANDUP":
                    # Only do support stand if currently sitting
                    rm.stand_from_sit_only()
                    last_cmd = "STOP"

                elif g in ("MOVELEFT", "TURNLEFT"):
                    rm.turn_left()
                    last_cmd = "TURN_LEFT"

                elif g in ("MOVERIGHT", "TURNRIGHT"):
                    rm.turn_right()
                    last_cmd = "TURN_RIGHT"

                elif g in ("BACK", "BACKWARD"):
                    rm.back()
                    last_cmd = "BACK"

                else:
                    rm.stop()
                    last_cmd = "STOP"

                map_server.update(points=pts, clearance=clr, cmd=last_cmd, robot_state=str(state))
                time.sleep(dt)
                continue

            # ===== 2) Auto move by lidar + camera =====
            lidar_dec = lidar_decision_label() or "FORWARD"
            lidar_dec = lidar_dec.upper().strip()
            if lidar_dec == "BACKWARD":
                lidar_dec = "BACK"

            if clr.get("FRONT", 9999.0) < EMERGENCY_STOP_CM:
                rm.stop()
                last_cmd = "STOP"
                map_server.update(points=pts, clearance=clr, cmd=last_cmd, robot_state=str(state))
                time.sleep(dt)
                continue

            prefer_forward = ("FORWARD" in lidar_dec or lidar_dec == "FORWARD")

            if prefer_forward and clr.get("FRONT", 0.0) >= SAFE_FORWARD_CM:
                if camera_is_clear():
                    cmd = "FORWARD"
                else:
                    cmd = pick_direction_by_clearance(clr, prefer="LEFT")
            else:
                if lidar_dec in ("TURN_LEFT", "TURN_RIGHT", "BACK", "STOP"):
                    cmd = lidar_dec
                else:
                    cmd = pick_direction_by_clearance(clr, prefer="FRONT")

            if not cmd:
                cmd = "STOP"

            # IMPORTANT: NO stabilize for FORWARD/TURN/BACK
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

            last_cmd = cmd
            map_server.update(points=pts, clearance=clr, cmd=last_cmd, robot_state=str(state))
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
