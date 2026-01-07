#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import wave
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from robot_hat import Servo, Music
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

WEB_PORT = 8000
CAM_DEV = os.environ.get("CAM_DEV", "0")
CAM_W = int(os.environ.get("CAM_W", "320"))
CAM_H = int(os.environ.get("CAM_H", "240"))
CAM_FPS = int(os.environ.get("CAM_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

# Boot lift angles (same flow as test_nanghaichansau.py)
REAR_LIFT_ANGLES = {
    "P4": 80,
    "P5": 30,
    "P6": -70,
    "P7": -30,
}

FRONT_LIFT_ANGLES = {
    "P0": -20,
    "P1": 90,
    "P2": 20,
    "P3": -75,
}

HEAD_INIT_ANGLES = {
    "P8": 28,
    "P9": -70,
    "P10": 90,
}


def clamp(angle: float) -> int:
    try:
        v = int(angle)
    except Exception:
        v = 0
    return max(-90, min(90, v))


def apply_angles(angles: dict[str, float], per_servo_delay: float = 0.03):
    for port, angle in angles.items():
        try:
            s = Servo(port)
            s.angle(clamp(angle))
        except Exception:
            pass
        time.sleep(per_servo_delay)

def set_servo_angle(port: str, angle: float, hold_sec: float = 0.4):
    try:
        s = Servo(port)
        s.angle(clamp(angle))
        time.sleep(max(0.05, float(hold_sec)))
        s.angle(clamp(angle))
    except Exception:
        pass


def smooth_pair(
    pA: str, a_start: int, a_end: int,
    pB: str, b_start: int, b_end: int,
    step: int = 1,
    delay: float = 0.03,
):
    sA = Servo(pA)
    sB = Servo(pB)

    a_start, a_end = clamp(a_start), clamp(a_end)
    b_start, b_end = clamp(b_start), clamp(b_end)

    a = a_start
    b = b_start

    try:
        sA.angle(a)
        sB.angle(b)
    except Exception:
        pass

    max_steps = max(abs(a_end - a_start), abs(b_end - b_start))
    if max_steps == 0:
        return

    step = max(1, int(abs(step)))

    for _ in range(max_steps):
        if a != a_end:
            a += step if a_end > a else -step
            if (a_end > a_start and a > a_end) or (a_end < a_start and a < a_end):
                a = a_end

        if b != b_end:
            b += step if b_end > b else -step
            if (b_end > b_start and b > b_end) or (b_end < b_start and b < b_end):
                b = b_end

        try:
            sA.angle(clamp(a))
            sB.angle(clamp(b))
        except Exception:
            pass

        time.sleep(delay)


def smooth_single(port: str, start: int, end: int, step: int = 1, delay: float = 0.03):
    s = Servo(port)
    start, end = clamp(start), clamp(end)
    a = start
    try:
        s.angle(a)
    except Exception:
        pass

    step = max(1, int(abs(step)))
    total = abs(end - start)
    if total == 0:
        return

    for _ in range(total):
        if a == end:
            break
        a += step if end > a else -step
        if (end > start and a > end) or (end < start and a < end):
            a = end
        try:
            s.angle(clamp(a))
        except Exception:
            pass
        time.sleep(delay)


def smooth_single_duration(port: str, start: int, end: int, duration_sec: float):
    total = abs(clamp(end) - clamp(start))
    if total == 0:
        return
    delay = max(0.01, float(duration_sec) / float(total))
    smooth_single(port, start, end, step=1, delay=delay)


def play_tiengsua(wav_path: str, volume: int = 80):
    if not os.path.exists(wav_path):
        print(f"[WARN] sound file not found: {wav_path}")
        return

    try:
        os.system("pinctrl set 12 op dh")
    except Exception:
        pass

    music = Music()
    try:
        music.music_set_volume(int(volume))
    except Exception:
        pass

    dur = None
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                dur = float(frames) / float(rate)
    except Exception:
        dur = None

    try:
        music.music_play(str(wav_path), loops=1)
    except Exception:
        return

    if dur is not None:
        time.sleep(dur + 0.15)
    else:
        time.sleep(2.5)


def _try_create_rgb_device():
    try:
        import robot_hat
    except Exception as e:
        print(f"[LED] robot_hat import failed: {e}")
        return None

    led_num = int(os.environ.get("PIDOG_LED_NUM", "2"))
    led_pin = int(os.environ.get("PIDOG_LED_PIN", "12"))
    candidates = (
        "RGBStrip",
        "RGBStripWS2812",
        "RGBStripAPA102",
        "RGBLed",
        "RGBLED",
    )
    arg_sets = (
        (),
        (led_num,),
        (led_num, led_pin),
        (led_pin, led_num),
    )

    for cls_name in candidates:
        cls = getattr(robot_hat, cls_name, None)
        if not cls:
            continue
        for args in arg_sets:
            try:
                dev = cls(*args)
                print(f"[LED] init {cls_name} args={args}")
                return dev
            except Exception:
                continue

    return None


def set_led(motion: MotionController, color: str, bps: float = 0.5):
    dog = getattr(motion, "dog", None)
    if not dog:
        print("[LED] motion has no dog instance")
        return

    rs = getattr(dog, "rgb_strip", None)
    rl = getattr(dog, "rgb_led", None)
    if not rs and not rl:
        dev = _try_create_rgb_device()
        if dev:
            try:
                dog.rgb_strip = dev
                rs = dev
            except Exception:
                pass
        else:
            print("[LED] no rgb device available (rgb_strip init failed?)")
            return

    # 1) rgb_strip.set_mode("breath", color, bps=?)
    try:
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


class BoardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # 0 empty, 1 player(O), 2 robot(|)

    def snapshot(self) -> List[List[int]]:
        with self._lock:
            return [row[:] for row in self._board]

    def set_player(self, r: int, c: int) -> bool:
        with self._lock:
            if self._board[r][c] == 0:
                self._board[r][c] = 1
                return True
            return False

    def set_robot(self, r: int, c: int) -> bool:
        with self._lock:
            if self._board[r][c] == 0:
                self._board[r][c] = 2
                return True
            return False

    def stats(self):
        with self._lock:
            empty = sum(1 for r in self._board for v in r if v == 0)
            player = sum(1 for r in self._board for v in r if v == 1)
            robot = sum(1 for r in self._board for v in r if v == 2)
        return {"empty": empty, "player": player, "robot": robot}

    def set_board(self, board: List[List[int]]):
        with self._lock:
            self._board = [row[:] for row in board]

    def first_empty(self) -> Tuple[int, int]:
        with self._lock:
            for r in range(3):
                for c in range(3):
                    if self._board[r][c] == 0:
                        return r, c
        return -1, -1


def detect_player_orange(frame_bgr, board: BoardState):
    h, w = frame_bgr.shape[:2]
    cell_w = w // 3
    cell_h = h // 3

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = (5, 80, 80)
    hi = (25, 255, 255)
    mask = cv2.inRange(hsv, lo, hi)

    for r in range(3):
        for c in range(3):
            y0 = r * cell_h
            x0 = c * cell_w
            roi = mask[y0:y0 + cell_h, x0:x0 + cell_w]
            if roi.size == 0:
                continue
            ratio = float(cv2.countNonZero(roi)) / float(roi.size)
            if ratio >= 0.02:
                board.set_player(r, c)


def detect_blue_lines(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if h < 40 or w < 40:
        return []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (90, 80, 60), (130, 255, 255))
    edges = cv2.Canny(blue_mask, 40, 120)
    min_len = int(min(w, h) * 0.20)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=25)
    if lines is None:
        return []

    out = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = x2 - x1
        dy = y2 - y1
        if (dx * dx + dy * dy) ** 0.5 < min_len:
            continue
        out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


def draw_board_overlay(frame_bgr, blue_lines=None):
    h, w = frame_bgr.shape[:2]
    if not blue_lines:
        return
    for x1, y1, x2, y2 in blue_lines:
        cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)


def _cluster_positions(pos, tol):
    if not pos:
        return []
    pos = sorted(pos)
    groups = [[pos[0]]]
    for p in pos[1:]:
        if abs(p - groups[-1][-1]) <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(sum(g) / len(g)) for g in groups]


def _pick_four(vals):
    if not vals:
        return []
    vals = sorted(vals)
    if len(vals) <= 4:
        return vals
    n = len(vals)
    return [vals[0], vals[(n - 1) // 3], vals[(2 * (n - 1)) // 3], vals[-1]]


def build_grid_from_lines(lines, frame_shape):
    h, w = frame_shape[:2]
    if not lines:
        return None
    v_pos = []
    h_pos = []
    for x1, y1, x2, y2 in lines:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < dy * 0.35:
            v_pos.append(int((x1 + x2) / 2))
        elif dy < dx * 0.35:
            h_pos.append(int((y1 + y2) / 2))
    tol = max(6, int(min(w, h) * 0.03))
    v_lines = _pick_four(_cluster_positions(v_pos, tol))
    h_lines = _pick_four(_cluster_positions(h_pos, tol))
    if len(v_lines) < 2 or len(h_lines) < 2:
        return None

    if len(v_lines) < 4:
        vmin, vmax = min(v_lines), max(v_lines)
        v_lines = [vmin, vmin + (vmax - vmin) // 3, vmin + 2 * (vmax - vmin) // 3, vmax]
    if len(h_lines) < 4:
        hmin, hmax = min(h_lines), max(h_lines)
        h_lines = [hmin, hmin + (hmax - hmin) // 3, hmin + 2 * (hmax - hmin) // 3, hmax]

    v_lines = sorted([max(0, min(w - 1, x)) for x in v_lines])
    h_lines = sorted([max(0, min(h - 1, y)) for y in h_lines])

    out_lines = []
    x0, x3 = v_lines[0], v_lines[-1]
    y0, y3 = h_lines[0], h_lines[-1]
    for x in v_lines:
        out_lines.append((x, y0, x, y3))
    for y in h_lines:
        out_lines.append((x0, y, x3, y))

    return {"x": v_lines, "y": h_lines, "lines": out_lines}


def detect_cell_states(frame_bgr, grid):
    x_lines = grid["x"]
    y_lines = grid["y"]
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, (5, 80, 80), (25, 255, 255))
    gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 40, 200))

    for r in range(3):
        for c in range(3):
            x0, x1 = x_lines[c], x_lines[c + 1]
            y0, y1 = y_lines[r], y_lines[r + 1]
            roi_o = orange_mask[y0:y1, x0:x1]
            roi_g = gray_mask[y0:y1, x0:x1]
            if roi_o.size == 0 or roi_g.size == 0:
                continue
            o_ratio = float(cv2.countNonZero(roi_o)) / float(roi_o.size)
            g_ratio = float(cv2.countNonZero(roi_g)) / float(roi_g.size)
            if o_ratio > 0.01:
                board[r][c] = 1
            elif g_ratio > 0.01:
                board[r][c] = 2
    return board


class CameraWeb:
    def __init__(self, board: BoardState):
        self.board = board
        self.app = Flask("tic_tac_toe_cam")
        self._lock = threading.Lock()
        self._last = None
        self._blue_lines = []
        self._p8_angle = 28
        self._p10_angle = 90
        self._lock_board_state = False
        self._stop = threading.Event()
        self._thread = None
        self._ready = threading.Event()
        self._failed = threading.Event()

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

        @self.app.get("/state.json")
        def state():
            st = self.board.stats()
            st["p8_angle"] = self._p8_angle
            st["p10_angle"] = self._p10_angle
            return jsonify(st)

        @self.app.get("/set_move")
        def set_move():
            who = str(request.args.get("who", "player"))
            r = int(request.args.get("r", "0"))
            c = int(request.args.get("c", "0"))
            ok = False
            if 0 <= r <= 2 and 0 <= c <= 2:
                if who == "robot":
                    ok = self.board.set_robot(r, c)
                else:
                    ok = self.board.set_player(r, c)
            return jsonify({"ok": ok})

        @self.app.get("/p8")
        def p8():
            action = str(request.args.get("action", ""))
            with self._lock:
                cur = int(self._p8_angle)
            if action == "inc":
                cur = min(90, cur + 1)
            elif action == "dec":
                cur = max(-90, cur - 1)
            elif action == "set":
                try:
                    cur = int(request.args.get("val", cur))
                except Exception:
                    pass
                cur = max(-90, min(90, cur))
            set_servo_angle("P8", cur, hold_sec=0.25)
            with self._lock:
                self._p8_angle = int(cur)
            return jsonify({"ok": True, "p8_angle": int(cur)})

        @self.app.get("/p10")
        def p10():
            action = str(request.args.get("action", ""))
            with self._lock:
                cur = int(self._p10_angle)
            if action == "inc":
                cur = min(90, cur + 1)
            elif action == "dec":
                cur = max(-90, cur - 1)
            elif action == "set":
                try:
                    cur = int(request.args.get("val", cur))
                except Exception:
                    pass
                cur = max(-90, min(90, cur))
            set_servo_angle("P10", cur, hold_sec=0.25)
            with self._lock:
                self._p10_angle = int(cur)
            return jsonify({"ok": True, "p10_angle": int(cur)})

        @self.app.get("/mjpeg")
        def mjpeg():
            return Response(self._mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _html(self) -> str:
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Ban Co Live</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }}
    .wrap {{ display:flex; gap:16px; padding:16px; align-items:flex-start; }}
    .card {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; min-width:260px; }}
    .kv {{ margin:8px 0; }}
    .k {{ color:#93c5fd; }}
    .row {{ display:flex; gap:8px; align-items:center; margin-top:10px; }}
    .btn {{ background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:6px 10px; border-radius:8px; cursor:pointer; }}
    .video {{ border:1px solid #223; border-radius:8px; width:{CAM_W}px; height:{CAM_H}px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="kv"><span class="k">Empty cells:</span> <span id="empty">-</span></div>
      <div class="kv"><span class="k">Robot moves:</span> <span id="robot">-</span></div>
      <div class="kv"><span class="k">Player moves:</span> <span id="player">-</span></div>
      <div class="kv"><span class="k">P8 angle:</span> <span id="p8">28</span></div>
      <div class="row">
        <button class="btn" onclick="p8Dec()">-</button>
        <button class="btn" onclick="p8Inc()">+</button>
      </div>
      <div class="kv" style="margin-top:12px;"><span class="k">P10 angle:</span> <span id="p10">90</span></div>
      <div class="row">
        <button class="btn" onclick="p10Dec()">-</button>
        <button class="btn" onclick="p10Inc()">+</button>
      </div>
      <div class="kv" style="font-size:12px;color:#aab;">Player = orange X, robot = short gray line</div>
    </div>
    <img class="video" id="cam" src="/mjpeg" />
  </div>
<script>
async function tick() {{
  try {{
    const r = await fetch('/state.json', {{cache:'no-store'}});
    const js = await r.json();
    document.getElementById('empty').textContent = js.empty ?? '-';
    document.getElementById('robot').textContent = js.robot ?? '-';
    document.getElementById('player').textContent = js.player ?? '-';
    document.getElementById('p8').textContent = js.p8_angle ?? '-';
    document.getElementById('p10').textContent = js.p10_angle ?? '-';
  }} catch(e) {{}}
}}
async function p8Inc() {{
  try {{ await fetch('/p8?action=inc'); }} catch(e) {{}}
  tick();
}}
async function p8Dec() {{
  try {{ await fetch('/p8?action=dec'); }} catch(e) {{}}
  tick();
}}
async function p10Inc() {{
  try {{ await fetch('/p10?action=inc'); }} catch(e) {{}}
  tick();
}}
async function p10Dec() {{
  try {{ await fetch('/p10?action=dec'); }} catch(e) {{}}
  tick();
}}
setInterval(tick, 500);
tick();
</script>
</body>
</html>"""

    def _mjpeg_gen(self):
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._last is None else self._last.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                time.sleep(0.03)
                continue
            jpg = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)

    def get_last_frame(self):
        with self._lock:
            return None if self._last is None else self._last.copy()

    def set_blue_lines(self, lines):
        with self._lock:
            self._blue_lines = lines or []

    def set_p8_angle(self, val: int):
        with self._lock:
            self._p8_angle = int(val)

    def set_p10_angle(self, val: int):
        with self._lock:
            self._p10_angle = int(val)

    def lock_board_state(self, on: bool = True):
        with self._lock:
            self._lock_board_state = bool(on)

    def _capture_loop(self):
        dev = int(CAM_DEV) if str(CAM_DEV).isdigit() else CAM_DEV
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[CAM] cannot open camera: {dev}")
            self._failed.set()
            self._ready.set()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            with self._lock:
                lock_board = self._lock_board_state
            if not lock_board:
                detect_player_orange(frame, self.board)
            blue_lines = detect_blue_lines(frame)
            self.set_blue_lines(blue_lines)
            draw_board_overlay(frame, blue_lines=blue_lines)

            with self._lock:
                self._last = frame
            self._ready.set()
            time.sleep(0.01)

        cap.release()

    def start(self):
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False, threaded=True),
            daemon=True,
        ).start()
        print(f"[WEB] http://<pi_ip>:{WEB_PORT}/", flush=True)

    def wait_ready(self, timeout_sec: float = 5.0) -> bool:
        self._ready.wait(timeout=max(0.1, float(timeout_sec)))
        return self._ready.is_set() and not self._failed.is_set()


def _detect_board_ready(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if h < 60 or w < 60:
        return False, None

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    white_ratio = float(cv2.countNonZero(white_mask)) / float(white_mask.size)

    blue_lines = detect_blue_lines(frame_bgr)
    if white_ratio < 0.10 or not blue_lines:
        return False, None

    return True, blue_lines


def searching_tictoeborad(cam: CameraWeb, motion: MotionController, timeout_sec: float = 60.0) -> bool:
    print("[SEARCH] start scanning for tictoe board")
    s8 = Servo("P8")

    try:
        s8.angle(clamp(28))
    except Exception:
        pass
    print("[SEARCH] P8 -> 28")

    t0 = time.time()
    while time.time() - t0 < float(timeout_sec):
        frame = cam.get_last_frame()
        if frame is not None:
            small = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
            ok, _lines = _detect_board_ready(small)
            if ok:
                print("[SEARCH] board ready")
                set_led(motion, "blue", bps=0.6)
                return True

        time.sleep(2.0)

    print("[SEARCH] timeout -> not found")
    set_led(motion, "red", bps=0.8)
    return False


def create_virtual_caroboard(cam: CameraWeb, motion: MotionController) -> bool:
    print("[VIRTUAL] start create_virtual_caroboard")
    lines_all = []

    def _scan(p8_angle: int):
        print(f"[VIRTUAL] P8 -> {p8_angle}")
        set_servo_angle("P8", p8_angle, hold_sec=0.4)
        cam.set_p8_angle(p8_angle)
        time.sleep(1.0)
        for ang in range(90, 19, -1):
            set_servo_angle("P10", ang, hold_sec=0.05)
            cam.set_p10_angle(ang)
            frame = cam.get_last_frame()
            if frame is None:
                continue
            small = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
            det = detect_blue_lines(small)
            if det:
                scale = 1.0 / 0.8
                for x1, y1, x2, y2 in det:
                    lines_all.append((int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)))

    _scan(15)
    time.sleep(2.0)
    _scan(38)

    if not lines_all:
        print("[VIRTUAL] no lines detected")
        set_led(motion, "red", bps=0.8)
        return False

    grid = build_grid_from_lines(lines_all, cam.get_last_frame() or np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8))
    if not grid:
        print("[VIRTUAL] cannot build grid from lines")
        set_led(motion, "red", bps=0.8)
        return False

    cam.set_blue_lines(grid["lines"])
    set_led(motion, "blue", bps=0.6)

    frame = cam.get_last_frame()
    if frame is not None:
        board = detect_cell_states(frame, grid)
        cam.board.set_board(board)
        cam.lock_board_state(True)
        print("[VIRTUAL] board updated")

    return True


def robotvehinhcaro():
    print("[PHASE 1] P2->24, P3->-90")
    smooth_single("P2", 0, 24, step=1, delay=0.03)
    smooth_single("P3", 0, -90, step=1, delay=0.03)

    print("[PHASE 2] P4->+35, P5->-43")
    smooth_single("P4", 0, 35, step=1, delay=0.03)
    smooth_single("P5", 0, -43, step=1, delay=0.03)

    print("[PHASE 3] P6->-22, P7->+39")
    smooth_single("P6", 0, -22, step=1, delay=0.03)
    smooth_single("P7", 0, 39, step=1, delay=0.03)

    print("[PHASE 3] wait 2s")
    time.sleep(2.0)

    print("[PHASE 4] P1->+73, P0: -69 -> +21 -> -69 (2s)")
    smooth_single("P1", 0, 73, step=1, delay=0.03)
    smooth_single("P0", 0, -69, step=1, delay=0.03)
    smooth_single("P0", -69, 21, step=1, delay=0.03)
    smooth_single_duration("P0", 21, -69, duration_sec=2.0)

    print("[PHASE] draw done")


def main():
    print("[START] test_dichuyentheobanco", flush=True)
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    os.environ.setdefault("PIDOG_SKIP_HEAD_INIT", "1")
    os.environ.setdefault("PIDOG_SKIP_MCU_RESET", "1")
    os.environ.setdefault("HEAD_P8_IDLE", "28")
    os.environ.setdefault("HEAD_SWEEP_MIN", "28")
    os.environ.setdefault("HEAD_SWEEP_MAX", "28")

    board = BoardState()
    cam = CameraWeb(board)
    cam.start()
    if not cam.wait_ready(timeout_sec=5.0):
        print("[CAM] not ready, stop", flush=True)
        return

    print("[BOOT] set P8 -> 28")
    set_servo_angle("P8", 28, hold_sec=0.4)
    print("[BOOT] set P10 -> 90")
    set_servo_angle("P10", 90, hold_sec=0.4)
    time.sleep(0.2)

    print("[BOOT] set head init angles")
    apply_angles(HEAD_INIT_ANGLES, per_servo_delay=0.04)
    print("[BOOT] head init done")

    print("[BOOT] lift rear legs (left then right)")
    smooth_pair("P4", 0, REAR_LIFT_ANGLES["P4"], "P5", 0, REAR_LIFT_ANGLES["P5"], step=1, delay=0.04)
    smooth_pair("P6", 0, REAR_LIFT_ANGLES["P6"], "P7", 0, REAR_LIFT_ANGLES["P7"], step=1, delay=0.04)
    print("[BOOT] rear lift done")
    time.sleep(2.0)

    print("[BOOT] lift front legs")
    apply_angles(FRONT_LIFT_ANGLES, per_servo_delay=0.04)
    print("[BOOT] front lift done")
    time.sleep(2.0)

    print("[BOOT] boot robot to stand")
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()
    print("[BOOT] boot done")
    motion.close()
    print("[BOOT] head controller stopped")

    play_tiengsua("tiengsua.wav")

    if not create_virtual_caroboard(cam, motion):
        print("[VIRTUAL] stop due to no board")
        dog = motion.get_dog()
        if dog:
            try:
                dog.do_action("stand", speed=5)
                dog.wait_all_done()
            except Exception:
                pass
        return

    print("[SEARCH] board ready -> keep web running")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
