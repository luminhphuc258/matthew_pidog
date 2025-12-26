#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import json
import random
import threading
import subprocess
import socket
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
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
SAFE_FORWARD_CM = 65.0     # muốn đi thẳng thì front clearance >=
SAFE_TURN_CM = 45.0        # rẽ trái/phải cần clearance >=
EMERGENCY_STOP_CM = 18.0   # quá gần thì stop ngay

# Periodic rest sit
REST_MIN_MINUTES = 2
REST_MAX_MINUTES = 5
REST_SIT_SECONDS = 30

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
    # giống code bạn (giữ nguyên)
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

def clamp(x, a, b):
    return a if x < a else (b if x > b else x)

# =========================
# Robot Status Manager
# =========================
class RobotState:
    """
    Lưu trạng thái để chuyển ổn định tránh té.
    """
    def __init__(self):
        self.mode = "BOOT"     # BOOT / STAND / SIT / MOVE / TURN / BACK / STOP
        self.detail = ""       # e.g. TURN_RIGHT
        self.ts = time.time()

    def set(self, mode: str, detail: str = ""):
        self.mode = mode
        self.detail = detail
        self.ts = time.time()

    def is_sitting(self) -> bool:
        return self.mode == "SIT"

    def is_standing(self) -> bool:
        return self.mode == "STAND"

    def __repr__(self):
        return f"{self.mode}:{self.detail}" if self.detail else self.mode

# =========================
# Motion wrapper (safe transitions)
# =========================
class RobotMotion:
    def __init__(self, motion: MotionController, pose_file: Path, state: RobotState):
        self.motion = motion
        self.pose_file = pose_file
        self.state = state
        self.dog = getattr(motion, "dog", None)

    def set_led(self, color: str, bps: float = 0.5):
        try:
            if self.dog and hasattr(self.dog, "rgb_strip"):
                self.dog.rgb_strip.set_mode("breath", color, bps=bps)
        except Exception:
            pass

    def _apply_pose_config(self):
        """
        Dùng y chang cách bạn đang xài (load_pose_config + apply_pose_from_cfg).
        """
        try:
            cfg = self.motion.load_pose_config()
            self.motion.apply_pose_from_cfg(cfg, per_servo_delay=0.02, settle_sec=0.6)
        except Exception:
            pass

    def boot_pose_stand(self):
        """
        THỨ TỰ BẮT BUỘC:
        1) boot
        2) pose chuẩn
        3) stand
        """
        self.state.set("BOOT")
        self.motion.boot()
        set_volumes()

        # pose chuẩn trước
        self._apply_pose_config()

        # rồi stand
        self.stop()
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
        Nếu đang SIT thì gọi support stand trước (để khỏi té),
        rồi mới apply pose và stand.
        """
        # cố gắng đứng dậy trước
        t0 = time.time()
        while time.time() - t0 < 0.7:
            try:
                self.motion.execute("STANDUP")
            except Exception:
                # fallback
                try:
                    self.motion.execute("STOP")
                except Exception:
                    pass
            time.sleep(0.02)

    def ensure_stand_stable(self):
        """
        Nếu đang sit -> support stand -> pose -> STOP(stand)
        """
        if self.state.is_sitting():
            self.support_stand_from_sit()

        # về pose chuẩn cho ổn định
        self._apply_pose_config()

        # stand
        try:
            self.motion.execute("STOP")
        except Exception:
            pass
        self.set_led("white", bps=0.35)
        self.state.set("STAND")

    def sit_down(self, seconds: float = REST_SIT_SECONDS):
        """
        Stop -> SIT -> giữ -> đứng lên -> pose -> stand
        """
        # stop trước
        try:
            self.motion.execute("STOP")
        except Exception:
            pass
        time.sleep(0.1)

        # sit
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

        # giữ 30s
        t_end = time.time() + float(seconds)
        while time.time() < t_end:
            time.sleep(0.1)

        # đứng lên ổn định lại
        set_face("what_is_it")
        self.ensure_stand_stable()

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
            # fallback
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
# Gesture Poller (highest priority)
# =========================
class GesturePoller:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_label = ""
        self.latest_ts = 0.0
        self.latest_raw = ""
        self.latest_face = ""

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
                raw = str(latest.get("raw", "") or "")
                face = str(latest.get("face", "") or "")

                with self.lock:
                    # chỉ update khi có label mới
                    if label:
                        self.latest_label = label
                        self.latest_ts = ts
                        self.latest_raw = raw
                        self.latest_face = face
            time.sleep(0.10)

    def get_active(self) -> Optional[str]:
        with self.lock:
            if not self.latest_label:
                return None
            if (time.time() - float(self.latest_ts)) > GESTURE_TTL_SEC:
                return None
            return self.latest_label

# =========================
# Lidar parsing + direction picking
# =========================
def _extract_points(scan_json: dict) -> List[Tuple[float, float]]:
    """
    Return list of (angle_deg, dist_cm)
    Tolerant với nhiều format.
    """
    if not isinstance(scan_json, dict):
        return []

    pts = []

    # common keys
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

    # arrays style: angles + dists
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
    """
    min dist in [a1..a2] degrees (handle wrap if needed)
    """
    if not points:
        return 9999.0

    def in_range(a):
        # normalize -180..180
        while a > 180:
            a -= 360
        while a < -180:
            a += 360
        return a

    a1n = in_range(a1)
    a2n = in_range(a2)

    mins = []
    for ang, dist in points:
        an = in_range(ang)
        d = float(dist)
        if d <= 0:
            continue
        if a1n <= a2n:
            if a1n <= an <= a2n:
                mins.append(d)
        else:
            # wrap
            if an >= a1n or an <= a2n:
                mins.append(d)

    return min(mins) if mins else 9999.0

def lidar_clearance(points: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Tính clearance (cm) theo sector.
    """
    return {
        "FRONT": _sector_min_distance(points, -25, 25),
        "LEFT":  _sector_min_distance(points, 30, 110),
        "RIGHT": _sector_min_distance(points, -110, -30),
        "BACK":  min(_sector_min_distance(points, 140, 180), _sector_min_distance(points, -180, -140)),
    }

def pick_direction_by_clearance(clr: Dict[str, float], prefer: str = "FRONT") -> str:
    """
    Chọn hướng tốt nhất.
    Return one of: FORWARD, TURN_LEFT, TURN_RIGHT, BACK, STOP
    """
    front = clr.get("FRONT", 0.0)
    left = clr.get("LEFT", 0.0)
    right = clr.get("RIGHT", 0.0)
    back = clr.get("BACK", 0.0)

    # emergency
    if min(front, left, right) < EMERGENCY_STOP_CM:
        # nếu quá sát nhiều phía -> lùi nhẹ
        if back > SAFE_TURN_CM:
            return "BACK"
        return "STOP"

    if prefer == "FRONT" and front >= SAFE_FORWARD_CM:
        return "FORWARD"

    # choose max clearance among left/right/front/back
    candidates = [
        ("FORWARD", front),
        ("TURN_LEFT", left),
        ("TURN_RIGHT", right),
        ("BACK", back),
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    best_cmd, best_val = candidates[0]

    # enforce minimums
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
        return True  # fail-open để robot vẫn có thể đi theo lidar
    label = str(js.get("label", "") or "").lower().strip()
    # service của bạn trả "no obstacle" / "yes have obstacle"
    return (label == "no obstacle")

# =========================
# Lidar status/decision
# =========================
def lidar_ready() -> bool:
    js = http_get_json(URL_LIDAR_STATUS, timeout=0.6)
    if not js:
        return False
    # tolerant keys
    for k in ("ok", "ready", "lidar_ready", "running"):
        if k in js:
            try:
                return bool(js[k])
            except Exception:
                pass
    # fallback: nếu có status string
    st = str(js.get("status", "") or "").lower()
    if "ready" in st or "running" in st:
        return True
    return False

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
    pts = _extract_points(js)
    return pts

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
            html = self._html()
            return Response(html, mimetype="text/html")

    def _html(self) -> str:
        # canvas "fake 3D": project polar -> x,y then perspective scaling
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

  // origin bottom-center for 3D feel
  const ox = cv.width * 0.5;
  const oy = cv.height * 0.78;

  // grid
  ctx.strokeStyle = '#142033';
  ctx.lineWidth = 1;
  for (let i=1;i<=6;i++) {{
    const rr = i * 70;
    ctx.beginPath();
    ctx.ellipse(ox, oy, rr, rr*0.55, 0, 0, Math.PI*2);
    ctx.stroke();
  }}

  // robot dot
  ctx.fillStyle = '#60a5fa';
  ctx.beginPath(); ctx.arc(ox, oy, 6, 0, Math.PI*2); ctx.fill();

  const pts = data.points || [];
  for (const p of pts) {{
    const ang = (p.angle || 0) * Math.PI/180.0;
    const d = (p.dist_cm || 0);
    if (!d || d <= 0) continue;

    // polar -> x,y (cm)
    const x = Math.sin(ang) * d;
    const y = Math.cos(ang) * d;

    // perspective: farther points compress upward
    const depth = Math.max(1, d);
    const scale = 1.0 / (1.0 + depth/260.0);

    const sx = ox + x * 1.3 * scale;
    const sy = oy - y * 1.1 * scale;

    // brightness: near = brighter
    const b = Math.max(40, Math.min(255, 300 - d));
    ctx.fillStyle = `rgb(${b},${b},${b})`;
    ctx.fillRect(sx, sy, 2, 2);
  }}

  // text overlays
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

    # motion
    state = RobotState()
    motion = MotionController(pose_file=POSE_FILE)
    rm = RobotMotion(motion=motion, pose_file=POSE_FILE, state=state)

    # map web
    map_server = MapServer()
    map_server.run_bg()

    # gesture poller
    gp = GesturePoller()
    gp.start()

    # BOOT -> POSE -> STAND (đúng thứ tự)
    rm.boot_pose_stand()
    rm.set_led("white", bps=0.35)
    set_face("what_is_it")

    # Wait lidar ready
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

    # rest scheduler
    next_rest_ts = time.time() + random.randint(REST_MIN_MINUTES * 60, REST_MAX_MINUTES * 60)

    # main loop
    dt = 1.0 / LOOP_HZ
    last_cmd = "STOP"

    print(f"[DEMO] map web: http://<pi_ip>:{MAP_PORT}/", flush=True)

    try:
        while True:
            now = time.time()

            # periodic sit rest
            if now >= next_rest_ts:
                rm.sit_down(seconds=REST_SIT_SECONDS)
                next_rest_ts = time.time() + random.randint(REST_MIN_MINUTES * 60, REST_MAX_MINUTES * 60)

            # get lidar scan
            pts = lidar_scan_points()
            clr = lidar_clearance(pts)

            # update map
            map_server.update(points=pts, clearance=clr, cmd=last_cmd, robot_state=str(state))

            # ===== 1) Gesture has highest priority =====
            g = gp.get_active()
            if g:
                g = g.upper().strip()
                # optional face
                # set_face("suprise")

                # gesture mapping
                if g in ("STOP", "STOPMUSIC"):
                    rm.stop()
                    last_cmd = "STOP"

                elif g == "SIT":
                    rm.sit_down(seconds=REST_SIT_SECONDS)
                    last_cmd = "SIT"

                elif g == "STANDUP":
                    rm.ensure_stand_stable()
                    last_cmd = "STAND"

                elif g in ("MOVELEFT", "TURNLEFT"):
                    rm.ensure_stand_stable()
                    rm.turn_left()
                    last_cmd = "TURN_LEFT"

                elif g in ("MOVERIGHT", "TURNRIGHT"):
                    rm.ensure_stand_stable()
                    rm.turn_right()
                    last_cmd = "TURN_RIGHT"

                elif g in ("BACK", "BACKWARD"):
                    rm.ensure_stand_stable()
                    rm.back()
                    last_cmd = "BACK"

                else:
                    # unknown gesture => stop for safety
                    rm.stop()
                    last_cmd = "STOP"

                time.sleep(dt)
                continue

            # ===== 2) Auto move by lidar + camera =====
            # - Nếu lidar nói forward clear -> check camera
            # - Nếu camera không clear -> chọn hướng khác dựa scan
            lidar_dec = lidar_decision_label()  # may be FORWARD / TURN_LEFT / TURN_RIGHT / STOP ...
            if not lidar_dec:
                lidar_dec = "FORWARD"

            lidar_dec = lidar_dec.upper().strip()
            if lidar_dec == "BACKWARD":
                lidar_dec = "BACK"

            # emergency stop if too close
            if clr.get("FRONT", 9999.0) < EMERGENCY_STOP_CM:
                rm.stop()
                last_cmd = "STOP"
                time.sleep(dt)
                continue

            # prefer by lidar
            prefer_cmd = "FORWARD" if "FORWARD" in lidar_dec or lidar_dec == "FORWARD" else None

            cmd = None

            if prefer_cmd == "FORWARD" and clr.get("FRONT", 0.0) >= SAFE_FORWARD_CM:
                # camera check
                if camera_is_clear():
                    cmd = "FORWARD"
                else:
                    # camera thấy obstacle => né theo lidar scan
                    cmd = pick_direction_by_clearance(clr, prefer="LEFT")  # ưu tiên né
            else:
                # lidar muốn rẽ hoặc stop
                if lidar_dec in ("TURN_LEFT", "TURN_RIGHT", "BACK", "STOP"):
                    cmd = lidar_dec
                else:
                    # fallback choose by clearance
                    cmd = pick_direction_by_clearance(clr, prefer="FRONT")

            if not cmd:
                cmd = "STOP"

            # ensure stable posture transitions
            if cmd in ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "BACK"):
                rm.ensure_stand_stable()

            # LED rules: forward blue, obstacle/avoid red
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
            # update map cmd immediately
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
