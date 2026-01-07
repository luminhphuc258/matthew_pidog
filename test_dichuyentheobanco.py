#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import wave
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from robot_hat import Servo
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

WEB_PORT = 8000
CAM_DEV = os.environ.get("CAM_DEV", "0")
CAM_W = int(os.environ.get("CAM_W", "320"))
CAM_H = int(os.environ.get("CAM_H", "240"))
CAM_FPS = int(os.environ.get("CAM_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))
ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")
GRID_COLS = int(os.environ.get("GRID_COLS", "6"))
GRID_ROWS = int(os.environ.get("GRID_ROWS", "4"))
WARP_SIZE = int(os.environ.get("WARP_SIZE", "600"))

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




def _run_cmd(cmd):
    try:
        return subprocess.run(cmd, check=False)
    except Exception:
        return None


def set_volumes():
    _run_cmd(["amixer", "-q", "sset", "robot-hat speaker", "100%"])
    _run_cmd(["amixer", "-q", "sset", "robot-hat speaker Playback Volume", "100%"])
    _run_cmd(["amixer", "-q", "sset", "robot-hat mic", "100%"])
    _run_cmd(["amixer", "-q", "sset", "robot-hat mic Capture Volume", "100%"])


def play_wav(path: str) -> bool:
    p = _run_cmd(["aplay", "-D", "default", "-q", path])
    return p is not None and p.returncode == 0


def play_tiengsua(wav_path: str, volume: int = 80):
    if not os.path.exists(wav_path):
        print(f"[WARN] sound file not found: {wav_path}")
        return

    try:
        os.system("pinctrl set 12 op dh")
    except Exception:
        pass

    set_volumes()

    dur = None
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                dur = float(frames) / float(rate)
    except Exception:
        dur = None

    if not play_wav(str(wav_path)):
        print("[WARN] aplay failed")
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
    def __init__(self, rows: int = GRID_ROWS, cols: int = GRID_COLS):
        self._lock = threading.Lock()
        self.rows = int(rows)
        self.cols = int(cols)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]  # 0 empty, 1 player(O), 2 robot(|)

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
            for r in range(self.rows):
                for c in range(self.cols):
                    if self._board[r][c] == 0:
                        return r, c
        return -1, -1


def detect_player_orange(frame_bgr, board: BoardState, grid=None):
    if not grid:
        return
    x_lines = grid.get("x") or []
    y_lines = grid.get("y") or []
    if len(x_lines) < board.cols + 1 or len(y_lines) < board.rows + 1:
        return

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = (5, 80, 80)
    hi = (25, 255, 255)
    mask = cv2.inRange(hsv, lo, hi)

    for r in range(board.rows):
        for c in range(board.cols):
            x0, x1 = x_lines[c], x_lines[c + 1]
            y0, y1 = y_lines[r], y_lines[r + 1]
            if x1 <= x0 or y1 <= y0:
                continue
            roi = mask[y0:y1, x0:x1]
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


def draw_board_overlay(frame_bgr, blue_lines=None, cells=None):
    if blue_lines:
        for x1, y1, x2, y2 in blue_lines:
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if cells:
        for cell in cells:
            x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 255), 1)






def _order_points(pts):
    pts = np.array(pts, dtype=float)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(diff))]
    bl = pts[int(np.argmax(diff))]
    return [tl, tr, br, bl]


def _find_board_rectangle(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if h < 60 or w < 60:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        area = abs(cv2.contourArea(approx))
        if area < (h * w * 0.08):
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)

    if best is None:
        return None
    return _order_points(best)


def _grid_lines_from_homography(grid):
    H = grid.get("H")
    if H is None:
        return []
    inv = np.linalg.inv(H)
    size = int(grid.get("warp_size", WARP_SIZE))
    x_lines = grid.get("x") or []
    y_lines = grid.get("y") or []
    lines = []
    for x in x_lines:
        src = np.array([[[x, 0]], [[x, size - 1]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, inv)
        x1, y1 = dst[0][0]
        x2, y2 = dst[1][0]
        lines.append((int(x1), int(y1), int(x2), int(y2)))
    for y in y_lines:
        src = np.array([[[0, y]], [[size - 1, y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, inv)
        x1, y1 = dst[0][0]
        x2, y2 = dst[1][0]
        lines.append((int(x1), int(y1), int(x2), int(y2)))
    return lines


def build_grid_from_rectangle(frame_bgr, cols: int = GRID_COLS, rows: int = GRID_ROWS):
    corners = _find_board_rectangle(frame_bgr)
    if not corners:
        return None
    dst = np.array([[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]], dtype=np.float32)
    src = np.array(corners, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    x_lines = _evenly_spaced(0, WARP_SIZE - 1, cols + 1)
    y_lines = _evenly_spaced(0, WARP_SIZE - 1, rows + 1)

    cells = []
    for r in range(rows):
        for c in range(cols):
            x_a, x_b = x_lines[c], x_lines[c + 1]
            y_a, y_b = y_lines[r], y_lines[r + 1]
            if x_b <= x_a or y_b <= y_a:
                continue
            cells.append({
                "r": r,
                "c": c,
                "x0": int(x_a),
                "y0": int(y_a),
                "x1": int(x_b),
                "y1": int(y_b),
                "cx": int((x_a + x_b) / 2),
                "cy": int((y_a + y_b) / 2),
            })

    grid = {"x": x_lines, "y": y_lines, "cells": cells, "rows": rows, "cols": cols, "H": H, "warp_size": WARP_SIZE}
    grid["lines"] = _grid_lines_from_homography(grid)
    return grid


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


def _pick_n(vals, n):
    if not vals:
        return []
    vals = sorted(vals)
    if len(vals) <= n:
        return vals
    if n == 1:
        return [vals[len(vals) // 2]]
    last = len(vals) - 1
    idxs = [int(round(i * last / (n - 1))) for i in range(n)]
    return [vals[i] for i in idxs]


def _evenly_spaced(min_v: int, max_v: int, count: int):
    if count <= 1:
        return [int(min_v)]
    if max_v <= min_v:
        return [int(min_v) for _ in range(count)]
    step = float(max_v - min_v) / float(count - 1)
    return [int(round(min_v + step * i)) for i in range(count)]


def build_grid_from_lines(lines, frame_shape, cols: int = GRID_COLS, rows: int = GRID_ROWS):
    if hasattr(frame_shape, 'shape'):
        h, w = frame_shape.shape[:2]
    else:
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
    v_lines = _pick_n(_cluster_positions(v_pos, tol), cols + 1)
    h_lines = _pick_n(_cluster_positions(h_pos, tol), rows + 1)
    if len(v_lines) < 2 or len(h_lines) < 2:
        return None

    if len(v_lines) < cols + 1:
        vmin, vmax = min(v_lines), max(v_lines)
        v_lines = _evenly_spaced(vmin, vmax, cols + 1)
    if len(h_lines) < rows + 1:
        hmin, hmax = min(h_lines), max(h_lines)
        h_lines = _evenly_spaced(hmin, hmax, rows + 1)

    v_lines = sorted([max(0, min(w - 1, x)) for x in v_lines])
    h_lines = sorted([max(0, min(h - 1, y)) for y in h_lines])

    out_lines = []
    x0, x3 = v_lines[0], v_lines[-1]
    y0, y3 = h_lines[0], h_lines[-1]
    for x in v_lines:
        out_lines.append((x, y0, x, y3))
    for y in h_lines:
        out_lines.append((x0, y, x3, y))

    cells = []
    for r in range(rows):
        for c in range(cols):
            x_a, x_b = v_lines[c], v_lines[c + 1]
            y_a, y_b = h_lines[r], h_lines[r + 1]
            if x_b <= x_a or y_b <= y_a:
                continue
            cells.append({
                "r": r,
                "c": c,
                "x0": int(x_a),
                "y0": int(y_a),
                "x1": int(x_b),
                "y1": int(y_b),
                "cx": int((x_a + x_b) / 2),
                "cy": int((y_a + y_b) / 2),
            })

    return {"x": v_lines, "y": h_lines, "lines": out_lines, "cells": cells, "rows": rows, "cols": cols}


def detect_cell_states(frame_bgr, grid, rows: int, cols: int):
    x_lines = grid["x"]
    y_lines = grid["y"]
    board = [[0 for _ in range(cols)] for _ in range(rows)]

    H = grid.get("H")
    size = int(grid.get("warp_size", WARP_SIZE))
    if H is None:
        return board
    warped = cv2.warpPerspective(frame_bgr, H, (size, size))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(hsv, (5, 80, 80), (25, 255, 255))
    gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 40, 200))

    for r in range(rows):
        for c in range(cols):
            x0, x1 = x_lines[c], x_lines[c + 1]
            y0, y1 = y_lines[r], y_lines[r + 1]
            if x1 <= x0 or y1 <= y0:
                continue
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
        self._grid = None
        self._cells = []
        self._p8_angle = 28
        self._p10_angle = 90
        self._lock_board_state = False
        self._stop = threading.Event()
        self._thread = None
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._play_requested = threading.Event()
        self._scan_status = "idle"

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

        @self.app.get("/state.json")
        def state():
            st = self.board.stats()
            st["p8_angle"] = self._p8_angle
            st["p10_angle"] = self._p10_angle
            with self._lock:
                st["grid_ok"] = self._grid is not None
                st["cells"] = self._cells
                st["scan_status"] = self._scan_status
            return jsonify(st)

        @self.app.get("/set_move")
        def set_move():
            who = str(request.args.get("who", "player"))
            r = int(request.args.get("r", "0"))
            c = int(request.args.get("c", "0"))
            ok = False
            if 0 <= r < self.board.rows and 0 <= c < self.board.cols:
                if who == "robot":
                    ok = self.board.set_robot(r, c)
                else:
                    ok = self.board.set_player(r, c)
            return jsonify({"ok": ok})

        @self.app.get("/play")
        def play():
            self.request_scan()
            return jsonify({"ok": True})

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
    .cells {{ font-size:11px; color:#cbd5f5; white-space:pre; max-height:260px; overflow:auto; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="kv"><span class="k">Empty cells:</span> <span id="empty">-</span></div>
      <div class="kv"><span class="k">Robot moves:</span> <span id="robot">-</span></div>
      <div class="kv"><span class="k">Player moves:</span> <span id="player">-</span></div>
      <div class="kv"><span class="k">Detected cells:</span> <span id="cells_count">-</span></div>
      <div id="cells" class="cells">-</div>
      <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">idle</span></div>
      <div class="row">
        <button class="btn" onclick="playScan()">Play</button>
      </div>
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
    const cells = js.cells || [];
    document.getElementById('cells_count').textContent = cells.length ?? '-';
    document.getElementById('cells').textContent = formatCells(cells);
    document.getElementById('p8').textContent = js.p8_angle ?? '-';
    document.getElementById('p10').textContent = js.p10_angle ?? '-';
    document.getElementById('scan_status').textContent = js.scan_status ?? '-';
  }} catch(e) {{}}
}}
function formatCells(cells) {{
  if (!cells || !cells.length) return '-';
  return cells.map(c => `(${{c.r}},${{c.c}}) [${{c.x0}},${{c.y0}}]-[${{c.x1}},${{c.y1}}]`).join('\n');
}}
async function playScan() {{
  try {{ await fetch('/play'); }} catch(e) {{}}
  tick();
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

    def set_grid(self, grid):
        with self._lock:
            self._grid = grid
            self._cells = [] if not grid else grid.get("cells", [])

    def set_p8_angle(self, val: int):
        with self._lock:
            self._p8_angle = int(val)

    def set_p10_angle(self, val: int):
        with self._lock:
            self._p10_angle = int(val)

    def request_scan(self):
        self._play_requested.set()

    def consume_scan_request(self) -> bool:
        if self._play_requested.is_set():
            self._play_requested.clear()
            return True
        return False

    def set_scan_status(self, status: str):
        with self._lock:
            self._scan_status = str(status)

    def lock_board_state(self, on: bool = True):
        with self._lock:
            self._lock_board_state = bool(on)

    def get_grid(self):
        with self._lock:
            return None if self._grid is None else dict(self._grid)

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
            grid = self.get_grid()
            if grid:
                if not lock_board:
                    board = detect_cell_states(frame, grid, self.board.rows, self.board.cols)
                    self.board.set_board(board)
                draw_board_overlay(frame, blue_lines=grid.get("lines"), cells=grid.get("cells"))
            else:
                if not lock_board:
                    detect_player_orange(frame, self.board, grid=None)
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
    print("[VIRTUAL] start create_virtual_caroboard (rectangle)")
    t0 = time.time()
    grid = None
    while time.time() - t0 < 8.0:
        frame = cam.get_last_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        grid = build_grid_from_rectangle(frame, cols=GRID_COLS, rows=GRID_ROWS)
        if grid:
            break
        time.sleep(0.1)

    if not grid:
        print("[VIRTUAL] cannot build grid from rectangle")
        set_led(motion, "red", bps=0.8)
        return False

    cam.set_grid(grid)
    set_led(motion, "blue", bps=0.6)

    frame = cam.get_last_frame()
    if frame is not None:
        board = detect_cell_states(frame, grid, GRID_ROWS, GRID_COLS)
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

    cam.set_scan_status("idle")
    print("[WEB] waiting for Play button to scan")
    try:
        while True:
            if cam.consume_scan_request():
                cam.set_scan_status("scanning")
                ok = create_virtual_caroboard(cam, motion)
                if ok:
                    cam.set_scan_status("ready")
                    print("[VIRTUAL] board ready -> keep web running")
                else:
                    cam.set_scan_status("failed")
                    print("[VIRTUAL] scan failed, press Play to retry")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C", flush=True)


if __name__ == "__main__":
    main()
