#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import wave
from pathlib import Path
from typing import List, Tuple

import cv2
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
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "0")).lower() in ("1", "true", "yes", "on")

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
    "P8": 80,
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


def draw_board_overlay(frame_bgr, board: BoardState):
    h, w = frame_bgr.shape[:2]
    cell_w = w // 3
    cell_h = h // 3

    overlay = frame_bgr.copy()

    # white fill for empty cells
    snap = board.snapshot()
    for r in range(3):
        for c in range(3):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = y0 + cell_h
            x1 = x0 + cell_w
            if snap[r][c] == 0:
                cv2.rectangle(overlay, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (255, 255, 255), -1)

    frame_bgr[:] = cv2.addWeighted(overlay, 0.18, frame_bgr, 0.82, 0)

    # grid lines
    grid_color = (160, 160, 160)
    for i in range(1, 3):
        cv2.line(frame_bgr, (i * cell_w, 0), (i * cell_w, h), grid_color, 2)
        cv2.line(frame_bgr, (0, i * cell_h), (w, i * cell_h), grid_color, 2)

    # pieces + borders
    for r in range(3):
        for c in range(3):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = y0 + cell_h
            x1 = x0 + cell_w
            cx = x0 + cell_w // 2
            cy = y0 + cell_h // 2
            if snap[r][c] == 1:
                cv2.circle(frame_bgr, (cx, cy), int(min(cell_w, cell_h) * 0.28), (0, 165, 255), 3)
                cv2.rectangle(frame_bgr, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 255, 0), 2)
            elif snap[r][c] == 2:
                cv2.line(frame_bgr, (cx - 10, cy), (cx + 10, cy), (120, 120, 120), 3)
                cv2.rectangle(frame_bgr, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (0, 0, 0), 2)


class CameraWeb:
    def __init__(self, board: BoardState):
        self.board = board
        self.app = Flask("tic_tac_toe_cam")
        self._lock = threading.Lock()
        self._last = None
        self._stop = threading.Event()
        self._thread = None

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

        @self.app.get("/state.json")
        def state():
            return jsonify(self.board.stats())

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
    .video {{ border:1px solid #223; border-radius:8px; width:{CAM_W}px; height:{CAM_H}px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="kv"><span class="k">Empty cells:</span> <span id="empty">-</span></div>
      <div class="kv"><span class="k">Robot moves:</span> <span id="robot">-</span></div>
      <div class="kv"><span class="k">Player moves:</span> <span id="player">-</span></div>
      <div class="kv" style="font-size:12px;color:#aab;">O = orange circle, robot = short gray line</div>
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
  }} catch(e) {{}}
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

    def _capture_loop(self):
        dev = int(CAM_DEV) if str(CAM_DEV).isdigit() else CAM_DEV
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[CAM] cannot open camera: {dev}")
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

            detect_player_orange(frame, self.board)
            draw_board_overlay(frame, self.board)

            with self._lock:
                self._last = frame
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
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    os.environ.setdefault("PIDOG_SKIP_HEAD_INIT", "1")
    os.environ.setdefault("PIDOG_SKIP_MCU_RESET", "1")

    board = BoardState()
    cam = CameraWeb(board)
    cam.start()

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

    play_tiengsua("tiengsua.wav")

    dog = motion.get_dog()
    if dog:
        r, c = board.first_empty()
        print(f"[MOVE] go to empty cell r={r} c={c}")
        try:
            dog.do_action("forward", speed=250)
            dog.wait_all_done()
        except Exception:
            pass
        board.set_robot(r, c)

        print("[MOVE] robot action")
        robotvehinhcaro()

        print("[MOVE] lift rear + front, then boot stand")
        smooth_pair("P4", 0, REAR_LIFT_ANGLES["P4"], "P5", 0, REAR_LIFT_ANGLES["P5"], step=1, delay=0.04)
        smooth_pair("P6", 0, REAR_LIFT_ANGLES["P6"], "P7", 0, REAR_LIFT_ANGLES["P7"], step=1, delay=0.04)
        apply_angles(FRONT_LIFT_ANGLES, per_servo_delay=0.04)
        motion.boot()

        print("[MOVE] backward 3s then stop")
        try:
            dog.do_action("backward", speed=250)
            time.sleep(3.0)
            dog.do_action("stand", speed=5)
            dog.wait_all_done()
        except Exception:
            pass

    print("[DONE]")


if __name__ == "__main__":
    main()
