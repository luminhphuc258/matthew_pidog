#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import json
import uuid
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file

from robot_hat import Servo
from motion_controller import MotionController


# =========================
# CONFIG
# =========================
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))

CAM_DEV = os.environ.get("CAM_DEV", "0")
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "12"))
JPEG_QUALITY = int(os.environ.get("CAM_JPEG_QUALITY", "70"))

ROTATE_180 = str(os.environ.get("CAM_ROTATE_180", "1")).lower() in ("1", "true", "yes", "on")

GRID_ROWS = int(os.environ.get("GRID_ROWS", "6"))
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))

CELL_PX = int(os.environ.get("CELL_PX", "90"))
WARP_W = GRID_COLS * CELL_PX
WARP_H = GRID_ROWS * CELL_PX

SCAN_API_URL = os.environ.get(
    "SCAN_API_URL",
    "https://embeddedprogramming-healtheworldserver.up.railway.app/scan_chess",
)
SCAN_TIMEOUT = float(os.environ.get("SCAN_TIMEOUT", "40"))

STATE_POLL_MS = int(os.environ.get("STATE_POLL_MS", "500"))
MJPEG_SLEEP = float(os.environ.get("MJPEG_SLEEP", "0.06"))

# ---------- Yellow marker HSV ----------
Y_H_MIN = int(os.environ.get("Y_H_MIN", "18"))
Y_H_MAX = int(os.environ.get("Y_H_MAX", "40"))
Y_S_MIN = int(os.environ.get("Y_S_MIN", "70"))
Y_S_MAX = int(os.environ.get("Y_S_MAX", "255"))
Y_V_MIN = int(os.environ.get("Y_V_MIN", "70"))
Y_V_MAX = int(os.environ.get("Y_V_MAX", "255"))

MARKER_MIN_AREA = int(os.environ.get("MARKER_MIN_AREA", "140"))
MARKER_MAX_AREA = int(os.environ.get("MARKER_MAX_AREA", "40000"))
MARKER_MAX_AR = float(os.environ.get("MARKER_MAX_AR", "6.0"))
MARKER_MIN_SIDE = int(os.environ.get("MARKER_MIN_SIDE", "8"))

CORNER_BAND = float(os.environ.get("CORNER_BAND", "0.22"))

# IMPORTANT: không còn reject top nữa, chỉ đánh dấu near_top để debug
TOP_NOISE_REJECT_Y = float(os.environ.get("TOP_NOISE_REJECT_Y", "0.10"))

BBOX_PAD_PX = int(os.environ.get("BBOX_PAD_PX", "0"))
ALLOW_INFER_4TH = str(os.environ.get("ALLOW_INFER_4TH", "1")).lower() in ("1", "true", "yes", "on")
BBOX_SMOOTH_ALPHA = float(os.environ.get("BBOX_SMOOTH_ALPHA", "0.25"))

TOP_X_MATCH_MAX = float(os.environ.get("TOP_X_MATCH_MAX", "0.22"))  # tolerance match theo x

LOG_KEEP = int(os.environ.get("LOG_KEEP", "250"))

# --- Preprocess grid lines (debug/optional) ---
# 0=off, 1=on
ENABLE_GRID_PREPROCESS = str(os.environ.get("ENABLE_GRID_PREPROCESS", "1")).lower() in ("1", "true", "yes", "on")
# adaptive threshold block size (odd)
GRID_ADAPT_BLOCK = int(os.environ.get("GRID_ADAPT_BLOCK", "21"))
GRID_ADAPT_C = int(os.environ.get("GRID_ADAPT_C", "7"))
# line kernel sizes (auto if <=0)
GRID_HK = int(os.environ.get("GRID_HK", "0"))
GRID_VK = int(os.environ.get("GRID_VK", "0"))
# thicken lines a bit
GRID_THICK_DILATE = int(os.environ.get("GRID_THICK_DILATE", "1"))  # 0..2
# remove tiny dots
GRID_DOT_OPEN = int(os.environ.get("GRID_DOT_OPEN", "1"))

# --- NEW: local OpenCV pipeline ---
GRID_SAVE_PATH = os.environ.get("GRID_SAVE_PATH", "grid_coords.json")
DESKEW_MAX_ANGLE = float(os.environ.get("DESKEW_MAX_ANGLE", "12.0"))
DESKEW_HOUGH_THRESH = int(os.environ.get("DESKEW_HOUGH_THRESH", "120"))
DESKEW_MIN_LINE = int(os.environ.get("DESKEW_MIN_LINE", "60"))
DESKEW_MAX_GAP = int(os.environ.get("DESKEW_MAX_GAP", "15"))

CELL_PAD_RATIO = float(os.environ.get("CELL_PAD_RATIO", "0.12"))
O_MIN_AREA_FRAC = float(os.environ.get("O_MIN_AREA_FRAC", "0.08"))
O_MAX_AREA_FRAC = float(os.environ.get("O_MAX_AREA_FRAC", "0.65"))
O_MIN_CIRC = float(os.environ.get("O_MIN_CIRC", "0.62"))
X_MIN_LINES = int(os.environ.get("X_MIN_LINES", "2"))
X_ANGLE_TOL = float(os.environ.get("X_ANGLE_TOL", "18.0"))
X_CENTER_RADIUS = float(os.environ.get("X_CENTER_RADIUS", "0.25"))

GRID_LINE_THICK = int(os.environ.get("GRID_LINE_THICK", "2"))
CELL_BORDER_THICK = int(os.environ.get("CELL_BORDER_THICK", "2"))
OVERLAY_ALPHA = float(os.environ.get("OVERLAY_ALPHA", "0.35"))

# --- Robot/arm config ---
POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

ARM_LIFT_PORT = os.environ.get("ARM_LIFT_PORT", "P11")
ARM_UP_ANGLE = int(os.environ.get("ARM_UP_ANGLE", "-77"))
ARM_DOWN_ANGLE = int(os.environ.get("ARM_DOWN_ANGLE", "-15"))
ARM_NEUTRAL_P10 = int(os.environ.get("ARM_NEUTRAL_P10", "85"))

ARM_P10_LEFT = int(os.environ.get("ARM_P10_LEFT", "90"))
ARM_P10_RIGHT = int(os.environ.get("ARM_P10_RIGHT", "41"))
ARM_P10_REVERSE = str(os.environ.get("ARM_P10_REVERSE", "0")).lower() in ("1", "true", "yes", "on")

ARM_P11_HOVER_MIN = int(os.environ.get("ARM_P11_HOVER_MIN", str(ARM_UP_ANGLE)))
ARM_P11_HOVER_MAX = int(os.environ.get("ARM_P11_HOVER_MAX", str(ARM_UP_ANGLE)))
ARM_TOUCH_HOLD_SEC = float(os.environ.get("ARM_TOUCH_HOLD_SEC", "0.5"))

PREPARE_INIT_P8 = int(os.environ.get("PREPARE_INIT_P8", "38"))
PREPARE_INIT_P10 = int(os.environ.get("PREPARE_INIT_P10", "75"))
PREPARE_HEAD_P8 = int(os.environ.get("PREPARE_HEAD_P8", "50"))

REAR_LIFT_ANGLES = {"P4": 80, "P5": 30, "P6": -70, "P7": -30}
FRONT_LIFT_ANGLES = {"P0": -20, "P1": 90, "P2": 20, "P3": -75}
HEAD_INIT_ANGLES = {"P8": 38, "P9": -70, "P10": 75}

POST_BOOT_P9 = int(os.environ.get("POST_BOOT_P9", "-76"))
POST_BOOT_P10 = int(os.environ.get("POST_BOOT_P10", "33"))
POST_BOOT_P11 = int(os.environ.get("POST_BOOT_P11", "-62"))

RESCAN_BACKWARD = str(os.environ.get("RESCAN_BACKWARD", "1")).lower() in ("1", "true", "yes", "on")
RESCAN_FORWARD = str(os.environ.get("RESCAN_FORWARD", "1")).lower() in ("1", "true", "yes", "on")


# =========================
# Helpers
# =========================
def now_s() -> str:
    return time.strftime("%H:%M:%S")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp_servo(angle: float) -> int:
    try:
        v = int(angle)
    except Exception:
        v = 0
    return max(-90, min(90, v))


def apply_angles(angles: Dict[str, float], per_servo_delay: float = 0.03):
    for port, angle in angles.items():
        try:
            s = Servo(port)
            s.angle(clamp_servo(angle))
        except Exception:
            pass
        time.sleep(per_servo_delay)


def set_servo_angle(port: str, angle: float, hold_sec: float = 0.4):
    try:
        s = Servo(port)
        s.angle(clamp_servo(angle))
        time.sleep(max(0.05, float(hold_sec)))
        s.angle(clamp_servo(angle))
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

    a_start, a_end = clamp_servo(a_start), clamp_servo(a_end)
    b_start, b_end = clamp_servo(b_start), clamp_servo(b_end)

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
            sA.angle(clamp_servo(a))
            sB.angle(clamp_servo(b))
        except Exception:
            pass

        time.sleep(delay)


def smooth_single(port: str, start: int, end: int, step: int = 1, delay: float = 0.03):
    s = Servo(port)
    start, end = clamp_servo(start), clamp_servo(end)
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
            s.angle(clamp_servo(a))
        except Exception:
            pass
        time.sleep(delay)


def smooth_single_duration(port: str, start: int, end: int, duration_sec: float):
    total = abs(clamp_servo(end) - clamp_servo(start))
    if total == 0:
        return
    delay = max(0.01, float(duration_sec) / float(total))
    smooth_single(port, start, end, step=1, delay=delay)


def _encode_jpeg(img_bgr, quality=JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return b""
    return buf.tobytes()


def _parse_json_response(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if text is None:
        return None, "empty response"
    raw = text.strip()
    if not raw:
        return None, "empty response"
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None, None
    except Exception as exc:
        pass

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else None, None
    except Exception as exc2:
        return None, str(exc2)

    return None, "parse failed"


def _post_image_to_api(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    if not image_bytes:
        return None

    boundary = uuid.uuid4().hex
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + image_bytes + footer

    req = urllib.request.Request(
        SCAN_API_URL,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=SCAN_TIMEOUT) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            parsed, perr = _parse_json_response(data)
            if parsed is None:
                return {"found": False, "error": f"parse failed: {perr}", "_raw": data}
            if "_raw" not in parsed:
                parsed["_raw"] = data
            return parsed
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        return {"found": False, "error": f"HTTPError {exc.code}: {detail[:300]}", "_raw": detail}
    except urllib.error.URLError as exc:
        return {"found": False, "error": f"URLError: {exc}"}
    except Exception as exc:
        return {"found": False, "error": f"parse failed: {exc}"}


def order_points_4(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def bbox_from_centers(centers: np.ndarray, w: int, h: int, pad_px: int = 0) -> Tuple[int, int, int, int]:
    xs = centers[:, 0]
    ys = centers[:, 1]
    x0 = int(np.floor(xs.min())) - pad_px
    x1 = int(np.ceil(xs.max())) + pad_px
    y0 = int(np.floor(ys.min())) - pad_px
    y1 = int(np.ceil(ys.max())) + pad_px
    x0 = clamp(x0, 0, w - 2)
    y0 = clamp(y0, 0, h - 2)
    x1 = clamp(x1, x0 + 2, w - 1)
    y1 = clamp(y1, y0 + 2, h - 1)
    return x0, y0, x1, y1


def warp_rect(frame_bgr, rect_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect_bbox
    src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))


def warp_perspective_from_centers(frame_bgr, centers4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = order_points_4(centers4)
    dst = np.array(
        [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(frame_bgr, M, (WARP_W, WARP_H))
    return warp, M


def smooth_bbox(prev: Optional[Tuple[int, int, int, int]], cur: Tuple[int, int, int, int], alpha: float) -> Tuple[int, int, int, int]:
    if prev is None:
        return cur
    px0, py0, px1, py1 = prev
    cx0, cy0, cx1, cy1 = cur
    x0 = int(round((1 - alpha) * px0 + alpha * cx0))
    y0 = int(round((1 - alpha) * py0 + alpha * cy0))
    x1 = int(round((1 - alpha) * px1 + alpha * cx1))
    y1 = int(round((1 - alpha) * py1 + alpha * cy1))
    return (x0, y0, x1, y1)


def prepare_robot(cam, robot_state: Dict[str, Any]):
    with robot_state["lock"]:
        if robot_state.get("prepared"):
            cam.set_prepare_status("ready")
            cam.log("[PREPARE] already prepared")
            return

    cam.set_prepare_status("preparing")
    cam.log("[PREPARE] boot to stand (same as test_dichuyentheobanco)")

    os.environ.setdefault("PIDOG_SKIP_HEAD_INIT", "1")
    os.environ.setdefault("PIDOG_SKIP_MCU_RESET", "1")
    os.environ.setdefault("HEAD_P8_IDLE", str(PREPARE_INIT_P8))
    os.environ.setdefault("HEAD_SWEEP_MIN", str(PREPARE_INIT_P8))
    os.environ.setdefault("HEAD_SWEEP_MAX", str(PREPARE_INIT_P8))

    try:
        cam.log("[PREPARE] set P8/P10 init")
        set_servo_angle("P8", PREPARE_INIT_P8, hold_sec=0.4)
        set_servo_angle("P10", PREPARE_INIT_P10, hold_sec=0.4)

        cam.log("[PREPARE] head init angles")
        apply_angles(HEAD_INIT_ANGLES, per_servo_delay=0.04)

        cam.log("[PREPARE] hold P7 at -30 during startup")
        set_servo_angle("P7", -30, hold_sec=0.4)

        cam.log("[PREPARE] lift rear legs")
        smooth_pair("P4", 0, REAR_LIFT_ANGLES["P4"], "P5", 0, REAR_LIFT_ANGLES["P5"], step=1, delay=0.04)
        smooth_pair("P6", 0, REAR_LIFT_ANGLES["P6"], "P7", 0, REAR_LIFT_ANGLES["P7"], step=1, delay=0.04)
        time.sleep(2.0)

        cam.log("[PREPARE] lift front legs")
        apply_angles(FRONT_LIFT_ANGLES, per_servo_delay=0.04)
        time.sleep(2.0)

        cam.log("[PREPARE] boot robot to stand")
        motion = MotionController(pose_file=POSE_FILE)
        motion.boot()
        try:
            motion.close()
        except Exception:
            pass

        time.sleep(2.0)
        cam.log("[PREPARE] force P8/P10 after MotionController")
        set_servo_angle("P8", 45, hold_sec=0.35)
        set_servo_angle("P10", 30, hold_sec=0.35)

        cam.log("[PREPARE] post-boot head/arm angles")
        set_servo_angle("P9", POST_BOOT_P9, hold_sec=0.35)
        set_servo_angle(ARM_LIFT_PORT, POST_BOOT_P11, hold_sec=0.35)

        cam.log("[PREPARE] arm up + head ready")
        set_servo_angle("P10", POST_BOOT_P10, hold_sec=0.35)

        with robot_state["lock"]:
            robot_state["motion"] = motion
            robot_state["prepared"] = True

        cam.set_prepare_status("ready")
        cam.log("[PREPARE] done")
    except Exception as exc:
        cam.set_prepare_status("failed")
        cam.log(f"[PREPARE] failed: {repr(exc)}")


def pick_empty_cell(board_mat: List[List[int]]) -> Optional[Tuple[int, int]]:
    for r, row in enumerate(board_mat):
        for c, v in enumerate(row):
            if v == 0:
                return r, c
    return None


def cell_center_px(row: int, col: int) -> Tuple[float, float]:
    return (col + 0.5) * float(CELL_PX), (row + 0.5) * float(CELL_PX)


def map_pixel_to_arm_angles(x_px: float, y_px: float) -> Tuple[int, int]:
    if WARP_W <= 1:
        x_norm = 0.5
    else:
        x_norm = clamp(float(x_px) / float(WARP_W - 1), 0.0, 1.0)
    if ARM_P10_REVERSE:
        x_norm = 1.0 - x_norm
    p10 = ARM_P10_LEFT + (ARM_P10_RIGHT - ARM_P10_LEFT) * x_norm

    if WARP_H <= 1:
        y_norm = 0.5
    else:
        y_norm = clamp(float(y_px) / float(WARP_H - 1), 0.0, 1.0)
    p11_hover = ARM_P11_HOVER_MIN + (ARM_P11_HOVER_MAX - ARM_P11_HOVER_MIN) * y_norm

    return int(round(p10)), int(round(p11_hover))


def move_arm_to_cell(cam, row: int, col: int):
    x_px, y_px = cell_center_px(row, col)
    p10, p11_hover = map_pixel_to_arm_angles(x_px, y_px)
    cam.log(f"[MOVE] cell=({row},{col}) center=({x_px:.1f},{y_px:.1f}) -> P10={p10} P11_hover={p11_hover}")

    time.sleep(2.0)
    for _ in range(2):
        smooth_single_duration(ARM_LIFT_PORT, ARM_UP_ANGLE, ARM_DOWN_ANGLE, duration_sec=1.0)
        smooth_single_duration(ARM_LIFT_PORT, ARM_DOWN_ANGLE, ARM_UP_ANGLE, duration_sec=1.0)


def perform_robot_move(cam, board: "BoardState", robot_state: Dict[str, Any], detector) -> None:
    with robot_state["lock"]:
        prepared = bool(robot_state.get("prepared"))
        motion = robot_state.get("motion")

    if not prepared:
        cam.log("[MOVE] skipped: robot not prepared")
        return

    board_mat = board.snapshot()
    target = pick_empty_cell(board_mat)
    if target is None:
        cam.log("[MOVE] no empty cell to play")
        return

    cam.log("[MOVE] start draw action")
    sel_poly = cam.get_cell_cam_poly(target[0], target[1])
    cam.set_selected_poly(sel_poly)
    move_arm_to_cell(cam, target[0], target[1])
    cam.set_selected_poly(None)
    cam.log("[MOVE] action done")


def _make_black_warp(text="NO_WARP") -> np.ndarray:
    img = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return img


def _angle_norm_deg(angle_deg: float) -> float:
    a = angle_deg % 180.0
    if a > 90.0:
        a -= 180.0
    return a


def deskew_warp(warp_bgr: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=DESKEW_HOUGH_THRESH,
        minLineLength=DESKEW_MIN_LINE,
        maxLineGap=DESKEW_MAX_GAP,
    )

    angles = []
    if lines is not None:
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            ang = _angle_norm_deg(ang)
            if abs(ang) <= DESKEW_MAX_ANGLE:
                angles.append(ang)
            elif abs(abs(ang) - 90.0) <= DESKEW_MAX_ANGLE:
                a2 = ang - 90.0 if ang > 0 else ang + 90.0
                angles.append(a2)

    angle = float(np.median(angles)) if angles else 0.0
    if abs(angle) > DESKEW_MAX_ANGLE:
        angle = 0.0

    h, w = warp_bgr.shape[:2]
    center = (w * 0.5, h * 0.5)
    R = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(warp_bgr, R, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return rotated, angle, R


def build_grid_lines(rows: int, cols: int, cell_px: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    lines = []
    w = cols * cell_px
    h = rows * cell_px
    for r in range(rows + 1):
        y = r * cell_px
        lines.append(((0, y), (w, y)))
    for c in range(cols + 1):
        x = c * cell_px
        lines.append(((x, 0), (x, h)))
    return lines


def draw_grid(img_bgr: np.ndarray, lines, color=(0, 255, 255), thick=2):
    for (x0, y0), (x1, y1) in lines:
        cv2.line(img_bgr, (int(x0), int(y0)), (int(x1), int(y1)), color, thick)


def overlay_frame(frame_bgr: np.ndarray, lines, polys, alpha: float) -> np.ndarray:
    out = frame_bgr.copy()
    if polys:
        overlay = out.copy()
        for poly in polys:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
        out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    if lines:
        draw_grid(out, lines, color=(0, 255, 255), thick=GRID_LINE_THICK)
    return out


def project_lines_to_camera(lines, M_inv: np.ndarray, invR: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    out = []
    for (x0, y0), (x1, y1) in lines:
        pts = np.array([[x0, y0], [x1, y1]], dtype=np.float32).reshape(-1, 1, 2)
        pts_warp = cv2.transform(pts, invR)
        pts_cam = cv2.perspectiveTransform(pts_warp, M_inv)
        p0 = pts_cam[0, 0]
        p1 = pts_cam[1, 0]
        out.append(((int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))))
    return out


def project_points_to_camera(pts, M_inv: np.ndarray, invR: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    pts_warp = cv2.transform(pts, invR)
    pts_cam = cv2.perspectiveTransform(pts_warp, M_inv)
    return pts_cam.reshape(-1, 2)


def save_grid_coords(path: str, data: Dict[str, Any]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
    except Exception:
        pass


def _line_intersection(p1, p2, p3, p4) -> Optional[Tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def detect_cell_state(cell_bgr: np.ndarray) -> int:
    h, w = cell_bgr.shape[:2]
    area = float(h * w)

    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # O detection (contour circularity)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < O_MIN_AREA_FRAC * area or a > O_MAX_AREA_FRAC * area:
            continue
        per = float(cv2.arcLength(c, True))
        if per <= 1.0:
            continue
        circ = 4.0 * np.pi * a / (per * per)
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / float(max(1, ch))
        if circ >= O_MIN_CIRC and 0.6 <= ar <= 1.4:
            return 2

    # X detection (two diagonal lines)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=max(10, int(0.5 * min(w, h))),
        maxLineGap=10,
    )
    if lines is None:
        return 0

    pos = []
    neg = []
    for l in lines[:, 0]:
        x1, y1, x2, y2 = l
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        ang = (ang + 180.0) % 180.0
        if abs(ang - 45.0) <= X_ANGLE_TOL:
            pos.append((x1, y1, x2, y2))
        elif abs(ang - 135.0) <= X_ANGLE_TOL:
            neg.append((x1, y1, x2, y2))

    if len(pos) + len(neg) < X_MIN_LINES:
        return 0

    cx, cy = w * 0.5, h * 0.5
    max_r = X_CENTER_RADIUS * min(w, h)
    for a in pos:
        for b in neg:
            p = _line_intersection((a[0], a[1]), (a[2], a[3]), (b[0], b[1]), (b[2], b[3]))
            if p is None:
                continue
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= max_r * max_r:
                return 1
    return 0


def detect_board_from_warp(warp_bgr: np.ndarray) -> Tuple[List[List[int]], np.ndarray, List[Dict[str, Any]]]:
    board = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    cells_info: List[Dict[str, Any]] = []

    overlay = warp_bgr.copy()
    pad = int(CELL_PAD_RATIO * CELL_PX)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x0 = c * CELL_PX
            y0 = r * CELL_PX
            x1 = x0 + CELL_PX
            y1 = y0 + CELL_PX

            ix0 = x0 + pad
            iy0 = y0 + pad
            ix1 = x1 - pad
            iy1 = y1 - pad
            ix0 = clamp(ix0, x0, x1 - 1)
            iy0 = clamp(iy0, y0, y1 - 1)
            ix1 = clamp(ix1, ix0 + 1, x1)
            iy1 = clamp(iy1, iy0 + 1, y1)

            cell = warp_bgr[iy0:iy1, ix0:ix1]
            state = detect_cell_state(cell)
            board[r][c] = state

            if state == 1:
                color = (0, 255, 255)  # yellow for player X
                label = "X"
            elif state == 2:
                color = (0, 0, 255)  # red for robot O
                label = "O"
            else:
                color = (160, 160, 160)  # gray for empty
                label = ""

            cv2.rectangle(overlay, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), color, CELL_BORDER_THICK)
            if label:
                cv2.putText(overlay, label, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cells_info.append({"row": r, "col": c, "state": "empty" if state == 0 else ("player_x" if state == 1 else "robot_o")})

    return board, overlay, cells_info

# =========================
# NEW: Grid preprocess (gray + remove dots + smooth + extract lines)
# =========================
def preprocess_grid_for_gpt(warp_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Output:
      - gray (uint8)
      - bin (0/255)
      - lines (0/255) : combined horizontal+vertical lines
      - final_bgr : white background + black clear grid lines (BGR)
    """
    h, w = warp_bgr.shape[:2]
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)

    # mild denoise (keeps edges)
    gray_blur = cv2.bilateralFilter(gray, 7, 50, 50)

    # adaptive threshold (invert so lines become white)
    blk = GRID_ADAPT_BLOCK
    if blk < 9:
        blk = 9
    if blk % 2 == 0:
        blk += 1

    bin_inv = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blk,
        GRID_ADAPT_C
    )

    # remove tiny dots
    if GRID_DOT_OPEN > 0:
        k = np.ones((3, 3), np.uint8)
        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k, iterations=GRID_DOT_OPEN)

    # line extraction kernels
    hk = GRID_HK if GRID_HK > 0 else max(15, w // 12)   # horizontal kernel length
    vk = GRID_VK if GRID_VK > 0 else max(15, h // 12)   # vertical kernel length

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    horiz = cv2.erode(bin_inv, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=1)

    vert = cv2.erode(bin_inv, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=1)

    lines = cv2.bitwise_or(horiz, vert)

    # connect broken segments a bit
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # thicken lines slightly
    if GRID_THICK_DILATE > 0:
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=GRID_THICK_DILATE)

    # final: white background with black grid lines
    final = 255 - lines  # lines black
    final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    # annotate for debug
    cv2.putText(final_bgr, "grid_clean_gray_lines", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return {
        "gray": gray,
        "bin_inv": bin_inv,
        "lines": lines,
        "final_bgr": final_bgr,
    }


# =========================
# Marker-only Detector (FIX: không reject top markers)
# =========================
class MarkerBoardDetector:
    def __init__(self):
        self.prev_bbox: Optional[Tuple[int, int, int, int]] = None

    def _mask_yellow(self, frame_bgr) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([Y_H_MIN, Y_S_MIN, Y_V_MIN], dtype=np.uint8)
        upper = np.array([Y_H_MAX, Y_S_MAX, Y_V_MAX], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _extract_candidates(self, frame_bgr) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        mask = self._mask_yellow(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cand: List[Dict[str, Any]] = []
        dbg = frame_bgr.copy()

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < MARKER_MIN_AREA or area > MARKER_MAX_AREA:
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            if rw < MARKER_MIN_SIDE or rh < MARKER_MIN_SIDE:
                continue

            ar = max(rw, rh) / max(1.0, min(rw, rh))
            if ar > MARKER_MAX_AR:
                continue

            near_top = (cy < TOP_NOISE_REJECT_Y * h)

            box = cv2.boxPoints(rect).astype(np.int32)

            # ✅ FIX: không reject nữa, chỉ đổi màu + label để debug
            if near_top:
                cv2.drawContours(dbg, [box], -1, (0, 165, 255), 2)  # orange
                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 165, 255), -1)
                cv2.putText(dbg, "NEAR_TOP", (int(cx) + 6, int(cy) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                cv2.drawContours(dbg, [box], -1, (0, 255, 0), 2)
                cv2.circle(dbg, (int(cx), int(cy)), 4, (0, 255, 0), -1)

            cand.append({
                "center": np.array([cx, cy], dtype=np.float32),
                "area": area,
                "ar": float(ar),
                "near_top": bool(near_top),
            })

        # annotate indices
        dbg2 = dbg.copy()
        for i, c in enumerate(cand):
            cx, cy = int(c["center"][0]), int(c["center"][1])
            tag = "T" if c.get("near_top") else " "
            cv2.putText(dbg2, f"#{i}{tag} ({cx},{cy})", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return {"mask": mask, "candidates": cand, "debug": dbg2}

    def _infer_missing_from_3(self, slots: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        keys = {"tl", "tr", "br", "bl"}
        missing = list(keys - set(slots.keys()))
        if len(missing) != 1:
            return None
        m = missing[0]

        tl = slots.get("tl")
        tr = slots.get("tr")
        br = slots.get("br")
        bl = slots.get("bl")

        if bl is not None and br is not None:
            v = br - bl
            if m == "tl" and tr is not None:
                slots["tl"] = tr - v
                return slots
            if m == "tr" and tl is not None:
                slots["tr"] = tl + v
                return slots

        if tl is not None and tr is not None:
            vt = tr - tl
            if m == "bl" and br is not None:
                slots["bl"] = br - vt
                return slots
            if m == "br" and bl is not None:
                slots["br"] = bl + vt
                return slots

        pts = list(slots.values())
        if len(pts) == 3:
            p0, p1, p2 = pts
            slots[m] = p0 + p2 - p1
            return slots
        return None

    def _pick_slots(self, candidates: List[Dict[str, Any]], w: int, h: int) -> Dict[str, Any]:
        if len(candidates) < 2:
            return {"slots": {}, "inferred": False}

        pts = np.array([c["center"] for c in candidates], dtype=np.float32)

        top_y = CORNER_BAND * h
        bot_y = (1.0 - CORNER_BAND) * h
        mid_x = 0.5 * w

        idx_all = list(range(len(pts)))
        idx_bottom = [i for i in idx_all if pts[i][1] >= bot_y]
        idx_top = [i for i in idx_all if pts[i][1] <= top_y]

        # fallback pools
        if len(idx_top) < 2:
            idx_top = sorted(idx_all, key=lambda i: float(pts[i][1]))[:max(2, min(6, len(idx_all)))]
        if len(idx_bottom) < 2:
            idx_bottom = sorted(idx_all, key=lambda i: float(pts[i][1]), reverse=True)[:max(2, min(6, len(idx_all)))]

        def nearest(pool, tx, ty, forbid=set()):
            best_i, best_d = None, 1e18
            for i in pool:
                if i in forbid:
                    continue
                dx = float(pts[i][0] - tx)
                dy = float(pts[i][1] - ty)
                d = dx*dx + dy*dy
                if d < best_d:
                    best_d = d
                    best_i = i
            return best_i

        used = set()
        slots: Dict[str, np.ndarray] = {}

        bl_i = nearest(idx_bottom, 0.0, float(h - 1), forbid=set())
        if bl_i is not None:
            slots["bl"] = pts[bl_i]; used.add(bl_i)

        br_i = nearest(idx_bottom, float(w - 1), float(h - 1), forbid=used)
        if br_i is not None:
            slots["br"] = pts[br_i]; used.add(br_i)

        max_dx = TOP_X_MATCH_MAX * w

        def best_top_match(pool, x_ref, forbid):
            best_i, best_s = None, 1e18
            for i in pool:
                if i in forbid:
                    continue
                x, y = float(pts[i][0]), float(pts[i][1])
                dx = abs(x - float(x_ref))
                if dx > max_dx:
                    continue
                s = (dx*dx) * 2.2 + (y*y) * 0.10
                if s < best_s:
                    best_s = s
                    best_i = i
            return best_i

        # TL x gần BL, TR x gần BR
        if "bl" in slots:
            tl_pool = [i for i in idx_top if pts[i][0] <= mid_x] or idx_top
            tl_i = best_top_match(tl_pool, slots["bl"][0], used)
            if tl_i is not None:
                slots["tl"] = pts[tl_i]; used.add(tl_i)

        if "br" in slots:
            tr_pool = [i for i in idx_top if pts[i][0] >= mid_x] or idx_top
            tr_i = best_top_match(tr_pool, slots["br"][0], used)
            if tr_i is not None:
                slots["tr"] = pts[tr_i]; used.add(tr_i)

        # fallback nếu chỉ thấy top
        if "tl" not in slots:
            tl_i2 = nearest(idx_top, 0.0, 0.0, used)
            if tl_i2 is not None:
                slots["tl"] = pts[tl_i2]; used.add(tl_i2)

        if "tr" not in slots:
            tr_i2 = nearest(idx_top, float(w - 1), 0.0, used)
            if tr_i2 is not None:
                slots["tr"] = pts[tr_i2]; used.add(tr_i2)

        inferred = False
        if ALLOW_INFER_4TH and len(slots) == 3:
            res = self._infer_missing_from_3(slots)
            if res is not None and len(res) == 4:
                inferred = True

        return {"slots": slots, "inferred": inferred}

    def detect_board_bbox(self, frame_bgr) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        ex = self._extract_candidates(frame_bgr)
        cand = ex["candidates"]
        dbg = ex["debug"].copy()
        mask = ex["mask"]

        cv2.putText(dbg, f"cand={len(cand)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)

        picked = self._pick_slots(cand, w, h)
        slots = picked["slots"]
        inferred = picked["inferred"]

        dbg_pick = dbg.copy()
        for k in ["tl", "tr", "br", "bl"]:
            if k in slots:
                p = slots[k]
                cv2.circle(dbg_pick, (int(p[0]), int(p[1])), 10, (255, 0, 255), -1)
                cv2.putText(dbg_pick, k.upper(), (int(p[0]) + 10, int(p[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        cv2.putText(dbg_pick, f"INFER={inferred}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        if len(slots) < 4:
            return {
                "found": False,
                "bbox": None,
                "mask": mask,
                "debug": dbg,
                "debug_pick": dbg_pick,
                "inferred": inferred,
                "slots": {k: [float(v[0]), float(v[1])] for k, v in slots.items()},
            }

        centers4 = np.array([slots["tl"], slots["tr"], slots["br"], slots["bl"]], dtype=np.float32)
        centers4 = order_points_4(centers4)

        bbox = bbox_from_centers(centers4, w, h, pad_px=BBOX_PAD_PX)
        bbox = smooth_bbox(self.prev_bbox, bbox, BBOX_SMOOTH_ALPHA)
        self.prev_bbox = bbox

        x0, y0, x1, y1 = bbox
        out = frame_bgr.copy()
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(out, "board_bbox (marker-only)", (x0 + 10, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return {
            "found": True,
            "bbox": bbox,
            "mask": mask,
            "debug": dbg,
            "debug_pick": dbg_pick,
            "board_bbox_img": out,
            "centers4": centers4,
            "inferred": inferred,
            "slots": {k: [float(v[0]), float(v[1])] for k, v in slots.items()},
        }


# =========================
# Board State store
# =========================
class BoardState:
    def __init__(self, rows: int, cols: int):
        self._lock = threading.Lock()
        self.rows = int(rows)
        self.cols = int(cols)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.last_scan_ts = 0.0

    def snapshot(self) -> List[List[int]]:
        with self._lock:
            return [row[:] for row in self._board]

    def set_board(self, board: List[List[int]]):
        with self._lock:
            self._board = [row[:] for row in board]
            self.last_scan_ts = time.time()

    def stats(self):
        with self._lock:
            empty = sum(1 for r in self._board for v in r if v == 0)
            player_x = sum(1 for r in self._board for v in r if v == 1)
            robot_o = sum(1 for r in self._board for v in r if v == 2)
            ts = self.last_scan_ts
        return {"empty": empty, "player_x": player_x, "robot_o": robot_o, "last_scan_ts": ts}


def board_from_server_cells(rows: int, cols: int, cells: List[Dict[str, Any]]) -> List[List[int]]:
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        r = int(cell.get("row", cell.get("r", -1)))
        c = int(cell.get("col", cell.get("c", -1)))
        st = str(cell.get("state", "empty")).lower().strip()
        if 0 <= r < rows and 0 <= c < cols:
            if st in ("player_x", "x", "human_x"):
                board[r][c] = 1
            elif st in ("robot_o", "robot_circle", "player_o", "o", "circle"):
                board[r][c] = 2
            else:
                board[r][c] = 0
    return board


# =========================
# Web + Camera + Logs
# =========================
class CameraWeb:
    def __init__(self, board: BoardState):
        self.board = board
        self.app = Flask("scan_board_marker_only")
        self._lock = threading.Lock()

        self._last_frame = None
        self._overlay_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self._overlay_x_polys: List[np.ndarray] = []
        self._overlay_labels: List[Tuple[Tuple[int, int], str, Tuple[int, int, int]]] = []
        self._selected_poly: Optional[np.ndarray] = None
        self._cell_cam_polys: Dict[Tuple[int, int], np.ndarray] = {}
        self._scan_status = "idle"
        self._prepare_status = "idle"
        self._last_error = ""
        self._cells_server: List[Dict[str, Any]] = []
        self._stages_jpg: Dict[str, bytes] = {}
        self._logs = deque(maxlen=LOG_KEEP)

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._play_requested = threading.Event()
        self._prepare_requested = threading.Event()

        @self.app.get("/")
        def index():
            return Response(self._html(), mimetype="text/html")

        @self.app.get("/state.json")
        def state():
            st = self.board.stats()
            with self._lock:
                st["scan_status"] = self._scan_status
                st["prepare_status"] = self._prepare_status
                st["last_error"] = self._last_error
                st["rows"] = self.board.rows
                st["cols"] = self.board.cols
                st["cells_server"] = self._cells_server
                st["stages"] = list(self._stages_jpg.keys())
            st["board"] = self.board.snapshot()
            return jsonify(st)

        @self.app.get("/logs.json")
        def logs():
            with self._lock:
                return jsonify({"lines": list(self._logs)})

        @self.app.get("/play")
        def play():
            self._play_requested.set()
            self.log("[UI] Play Scan clicked")
            return jsonify({"ok": True})

        @self.app.get("/prepare")
        def prepare():
            self._prepare_requested.set()
            self.log("[UI] Prepare clicked")
            return jsonify({"ok": True})

        @self.app.get("/mjpeg")
        def mjpeg():
            return Response(self._mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.get("/stage/<name>.jpg")
        def stage(name):
            with self._lock:
                data = self._stages_jpg.get(name)
            if not data:
                return ("not found", 404)
            return send_file(BytesIO(data), mimetype="image/jpeg")

    def log(self, msg: str):
        line = f"{now_s()} {msg}"
        with self._lock:
            self._logs.append(line)
        print(line, flush=True)

    def _html(self) -> str:
        html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Board 6x4 (marker-only)</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b0f14; color:#e7eef7; margin:0; }}
    .wrap {{ display:grid; grid-template-columns: 1fr 420px; gap:14px; padding:14px; }}
    .left {{ display:flex; flex-direction:column; gap:14px; align-items:stretch; }}
    .card {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; min-width:360px; max-height: 45vh; overflow:auto; }}
    .kv {{ margin:8px 0; }}
    .k {{ color:#93c5fd; }}
    .err {{ color:#fca5a5; font-size:12px; white-space:pre-wrap; }}
    .btn {{ background:#1f2937; border:1px solid #334155; color:#e7eef7; padding:8px 12px; border-radius:10px; cursor:pointer; }}
    .video {{ border:1px solid #223; border-radius:10px; width:100%; max-width:{cam_w}px; height:auto; background:#000; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre; font-size:12px; color:#cbd5f5; }}
    .stages {{ background:#111827; border:1px solid #223; border-radius:12px; padding:12px; max-height: calc(100vh - 28px); overflow:auto; }}
    .stageimg {{ width:100%; border-radius:10px; border:1px solid #223; margin-top:10px; }}
    .title {{ font-weight:700; margin-bottom:8px; }}
    .muted {{ color:#93a4b8; font-size:12px; }}
    .logs {{ background:#0f172a; border:1px solid #223; border-radius:10px; padding:10px; margin-top:10px; max-height:160px; overflow:auto; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="left">
      <div class="card">
        <div class="kv"><span class="k">Board:</span> rows=<span id="rows">-</span>, cols=<span id="cols">-</span></div>
        <div class="kv"><span class="k">Scan status:</span> <span id="scan_status">-</span></div>
        <div class="kv"><span class="k">Prepare status:</span> <span id="prepare_status">-</span></div>
        <div class="kv"><span class="k">Last error:</span></div>
        <div id="last_error" class="err">-</div>

        <div class="kv"><span class="k">Empty:</span> <span id="empty">-</span></div>
        <div class="kv"><span class="k">Player X:</span> <span id="player_x">-</span></div>
        <div class="kv"><span class="k">Robot O:</span> <span id="robot_o">-</span></div>

        <div class="kv"><span class="k">Board state:</span></div>
        <div id="board" class="mono">-</div>

        <div class="kv"><span class="k">Cells (server):</span></div>
        <div id="cells" class="mono" style="max-height:160px; overflow:auto;">-</div>

        <div class="kv">
          <button class="btn" onclick="prepareRobot()">Prepare to go</button>
          <button class="btn" onclick="playScan()">Play Scan</button>
          <div class="muted" style="margin-top:8px;">NEW: perspective + deskew + server detect X.</div>
        </div>

        <div class="kv"><span class="k">Logs:</span></div>
        <div id="logs" class="logs mono">-</div>
      </div>

      <img class="video" id="cam" src="/mjpeg" />
    </div>

    <div class="stages">
      <div class="title">Processing Stages</div>
      <div class="muted">Always show candidates/picked/warp/deskew/grid/cells overlay.</div>
      <div id="stage_list"></div>
    </div>
  </div>

<script>
let last_ts = 0;

function formatBoard(board) {{
  if (!board || !board.length) return '-';
  return board.map(row => row.join('')).join('\\n');
}}

function formatCells(cells) {{
  if (!cells || !cells.length) return '-';
  return cells.map(c => `(${{c.row}},${{c.col}}) ${{c.state}}`).join('\\n');
}}

function renderStages(names) {{
  const host = document.getElementById('stage_list');
  host.innerHTML = '';
  (names || []).forEach(n => {{
    const t = document.createElement('div');
    t.className = 'muted';
    t.textContent = n;
    const img = document.createElement('img');
    img.className = 'stageimg';
    img.src = `/stage/${{n}}.jpg?ts=${{Date.now()}}`;
    host.appendChild(t);
    host.appendChild(img);
  }});
}}

async function tick() {{
  try {{
    const r = await fetch('/state.json', {{cache:'no-store'}});
    const js = await r.json();

    document.getElementById('rows').textContent = js.rows ?? '-';
    document.getElementById('cols').textContent = js.cols ?? '-';
    document.getElementById('scan_status').textContent = js.scan_status ?? '-';
    document.getElementById('prepare_status').textContent = js.prepare_status ?? '-';
    document.getElementById('last_error').textContent = js.last_error || '-';

    document.getElementById('empty').textContent = js.empty ?? '-';
    document.getElementById('player_x').textContent = js.player_x ?? '-';
    document.getElementById('robot_o').textContent = js.robot_o ?? '-';

    document.getElementById('board').textContent = formatBoard(js.board);
    document.getElementById('cells').textContent = formatCells(js.cells_server);

    const ts = js.last_scan_ts || 0;
    if (ts > 0 && ts !== last_ts) {{
      last_ts = ts;
      renderStages(js.stages || []);
    }}

    const lr = await fetch('/logs.json', {{cache:'no-store'}});
    const ljs = await lr.json();
    document.getElementById('logs').textContent = (ljs.lines || []).slice(-120).join('\\n') || '-';
  }} catch(e) {{}}
}}

async function playScan() {{
  try {{ await fetch('/play'); }} catch(e) {{}}
  tick();
}}

async function prepareRobot() {{
  try {{ await fetch('/prepare'); }} catch(e) {{}}
  tick();
}}

setInterval(tick, {poll_ms});
tick();
</script>
</body>
</html>
"""
        return html.format(cam_w=CAM_W, cam_h=CAM_H, poll_ms=STATE_POLL_MS)

    def _mjpeg_gen(self):
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._last_frame is None else self._last_frame.copy()
                lines = list(self._overlay_lines)
                polys = list(self._overlay_x_polys)
                labels = list(self._overlay_labels)
                selected = None if self._selected_poly is None else self._selected_poly.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            if lines or polys:
                frame = overlay_frame(frame, lines, polys, OVERLAY_ALPHA)
            if selected is not None:
                overlay = frame.copy()
                pts = np.asarray(selected, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            if labels:
                for (x, y), text, color in labels:
                    cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            jpg = _encode_jpeg(frame, quality=JPEG_QUALITY)
            if not jpg:
                time.sleep(0.03)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(MJPEG_SLEEP)

    def get_last_frame(self):
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def set_last_frame(self, frame):
        with self._lock:
            self._last_frame = frame

    def set_scan_status(self, status: str, error: str = ""):
        with self._lock:
            self._scan_status = str(status)
            self._last_error = str(error or "")

    def set_prepare_status(self, status: str):
        with self._lock:
            self._prepare_status = str(status)

    def set_overlay(self, lines, x_polys):
        with self._lock:
            self._overlay_lines = list(lines or [])
            self._overlay_x_polys = list(x_polys or [])

    def set_overlay_labels(self, labels):
        with self._lock:
            self._overlay_labels = list(labels or [])

    def set_selected_poly(self, poly: Optional[np.ndarray]):
        with self._lock:
            self._selected_poly = None if poly is None else np.asarray(poly, dtype=np.float32)

    def set_cell_cam_polys(self, polys: Dict[Tuple[int, int], np.ndarray]):
        with self._lock:
            self._cell_cam_polys = dict(polys or {})

    def get_cell_cam_poly(self, row: int, col: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._cell_cam_polys.get((row, col))

    def clear_overlay(self):
        with self._lock:
            self._overlay_lines = []
            self._overlay_x_polys = []
            self._overlay_labels = []
            self._selected_poly = None
            self._cell_cam_polys = {}

    def set_cells_server(self, cells: List[Dict[str, Any]]):
        with self._lock:
            self._cells_server = cells or []

    def set_stage_images(self, stages: Dict[str, np.ndarray]):
        jpgs = {}
        for name, img in stages.items():
            if img is None:
                continue
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img
            jpgs[name] = _encode_jpeg(img_bgr, quality=80)
        with self._lock:
            self._stages_jpg = jpgs

    def consume_scan_request(self) -> bool:
        if self._play_requested.is_set():
            self._play_requested.clear()
            self.clear_overlay()
            return True
        return False

    def consume_prepare_request(self) -> bool:
        if self._prepare_requested.is_set():
            self._prepare_requested.clear()
            return True
        return False

    def _draw_overlay(self, frame, lines, polys):
        return overlay_frame(frame, lines, polys, OVERLAY_ALPHA)

    def _capture_loop(self):
        dev = int(CAM_DEV) if str(CAM_DEV).isdigit() else CAM_DEV
        self.log(f"[CAM] opening {dev}")
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        if not cap.isOpened():
            self.log(f"[CAM] cannot open camera: {dev}")
            self._failed.set()
            self._ready.set()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.log(f"[CAM] opened ok, {CAM_W}x{CAM_H} fps={CAM_FPS}")

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            self.set_last_frame(frame)
            self._ready.set()
            time.sleep(0.01)

        cap.release()

    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(
            target=lambda: self.app.run(
                host="0.0.0.0",
                port=WEB_PORT,
                debug=False,
                use_reloader=False,
                threaded=True,
            ),
            daemon=True,
        ).start()
        self.log(f"[WEB] http://<pi_ip>:{WEB_PORT}/")

    def wait_ready(self, timeout_sec: float = 5.0) -> bool:
        self._ready.wait(timeout=max(0.1, float(timeout_sec)))
        return self._ready.is_set() and not self._failed.is_set()


# =========================
# Scan pipeline
# =========================
def run_scan_pipeline(cam: CameraWeb, detector: MarkerBoardDetector, board: "BoardState") -> Dict[str, Any]:
    cam.log("[SCAN] start")
    cam.set_scan_status("scanning", "")

    stages: Dict[str, np.ndarray] = {}
    try:
        frame = cam.get_last_frame()
        if frame is None:
            stages["0_fail"] = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            cam.set_stage_images(stages)
            cam.set_cells_server([])
            cam.set_scan_status("failed", "no frame from camera")
            cam.log("[SCAN] failed: no frame")
            return {"ok": False}

        det = detector.detect_board_bbox(frame)
        stages["3m_marker_mask"] = det.get("mask")
        stages["3m_candidates_debug"] = det.get("debug")
        stages["3m_picked_debug"] = det.get("debug_pick", det.get("debug"))

        board_bbox_img = det.get("board_bbox_img", stages["3m_picked_debug"])
        stages["4_board_bbox"] = board_bbox_img

        if det.get("found") and det.get("bbox") is not None and det.get("centers4") is not None:
            centers4 = det.get("centers4")
            warp_persp, M = warp_perspective_from_centers(frame, centers4)
            stages["5_warp_persp"] = warp_persp

            warp_deskew, angle, R = deskew_warp(warp_persp)
            stages["5e_deskew"] = warp_deskew

            grid_lines = build_grid_lines(GRID_ROWS, GRID_COLS, CELL_PX)
            grid_on_warp = warp_deskew.copy()
            draw_grid(grid_on_warp, grid_lines, color=(0, 255, 255), thick=GRID_LINE_THICK)
            stages["6_grid_on_warp"] = grid_on_warp

            M_inv = np.linalg.inv(M)
            invR = cv2.invertAffineTransform(R)
            cam_overlay = frame.copy()
            cam_lines = project_lines_to_camera(grid_lines, M_inv, invR)
            draw_grid(cam_overlay, cam_lines, color=(0, 255, 255), thick=GRID_LINE_THICK)

            cam_cell_polys: Dict[Tuple[int, int], np.ndarray] = {}
            cam_labels: List[Tuple[Tuple[int, int], str, Tuple[int, int, int]]] = []
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    x0 = c * CELL_PX
                    y0 = r * CELL_PX
                    x1 = x0 + CELL_PX
                    y1 = y0 + CELL_PX
                    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    cam_pts = project_points_to_camera(pts, M_inv, invR)
                    cam_cell_polys[(r, c)] = cam_pts

                    cx, cy = cell_center_px(r, c)
                    p10, p11 = map_pixel_to_arm_angles(cx, cy)
                    label = f"{p10},{p11}"
                    ccam = project_points_to_camera([(cx, cy)], M_inv, invR)
                    lx, ly = ccam[0]
                    cam_labels.append(((int(lx) - 12, int(ly) + 4), label, (0, 200, 255)))

            coords = {
                "rows": GRID_ROWS,
                "cols": GRID_COLS,
                "cell_px": CELL_PX,
                "warp_size": [WARP_W, WARP_H],
                "deskew_angle": float(angle),
                "markers": det.get("slots", {}),
                "warp_lines": [([int(a[0]), int(a[1])], [int(b[0]), int(b[1])]) for a, b in grid_lines],
                "camera_lines": [([int(a[0]), int(a[1])], [int(b[0]), int(b[1])]) for a, b in cam_lines],
            }
            save_grid_coords(GRID_SAVE_PATH, coords)

            cam.log("[SCAN] posting to server (grid_on_warp)...")
            result = _post_image_to_api(_encode_jpeg(grid_on_warp, quality=80))
            raw_resp = (result or {}).get("_raw", "")
            if raw_resp:
                cam.log(f"[SERVER] raw: {raw_resp[:400]}")

            if not result or not result.get("found"):
                err = (result or {}).get("error", "server returned found=false")
                cam.set_stage_images(stages)
                cam.set_cells_server([])
                cam.set_scan_status("failed", f"server fail: {err}")
                cam.clear_overlay()
                cam.log(f"[SCAN] failed: server: {err}")
                return {"ok": False}

            cells = result.get("cells", []) or []
            cam.set_cells_server(cells)
            board_mat = board_from_server_cells(GRID_ROWS, GRID_COLS, cells)
            board.set_board(board_mat)

            x_polys = []
            x_polys_warp = []
            for cell in cells:
                r = int(cell.get("row", cell.get("r", -1)))
                c = int(cell.get("col", cell.get("c", -1)))
                st = str(cell.get("state", "empty")).lower().strip()
                if st not in ("player_x", "x", "human_x"):
                    continue
                if r < 0 or c < 0 or r >= GRID_ROWS or c >= GRID_COLS:
                    continue
                x0 = c * CELL_PX
                y0 = r * CELL_PX
                x1 = x0 + CELL_PX
                y1 = y0 + CELL_PX
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                x_polys_warp.append(np.asarray(pts, dtype=np.float32))
                cam_pts = project_points_to_camera(pts, M_inv, invR)
                x_polys.append(cam_pts)

            cam.set_overlay(cam_lines, x_polys)
            cam.set_overlay_labels(cam_labels)
            cam.set_cell_cam_polys(cam_cell_polys)
            stages["6_grid_on_cam"] = overlay_frame(cam_overlay, cam_lines, x_polys, OVERLAY_ALPHA)
            stages["6a_grid_x_warp"] = overlay_frame(grid_on_warp, grid_lines, x_polys_warp, OVERLAY_ALPHA)
        else:
            stages["5_warp_persp"] = _make_black_warp("NO_WARP (need >=3 markers)")

        if not det.get("found") or det.get("bbox") is None or det.get("centers4") is None:
            cam.set_stage_images(stages)
            cam.set_cells_server([])
            cam.set_scan_status("failed", "board corners not found (need >=3 markers)")
            cam.clear_overlay()
            cam.log(f"[SCAN] failed: corners not found, slots={det.get('slots')}")
            return {"ok": False}

        cam.set_stage_images(stages)
        cam.set_scan_status("ready", "")
        cam.log(f"[SCAN] ok: server detect bbox={det['bbox']}")
        return {"ok": True}

    except Exception as e:
        cam.set_stage_images(stages or {})
        cam.set_cells_server([])
        cam.set_scan_status("failed", f"exception: {repr(e)}")
        cam.log(f"[SCAN] EXCEPTION: {repr(e)}")
        return {"ok": False}


# =========================
# MAIN
# =========================
def main():
    print("[START] marker_only_board_bbox_send_bbox_and_warp", flush=True)
    board = BoardState(rows=GRID_ROWS, cols=GRID_COLS)
    cam = CameraWeb(board)
    cam.start()

    if not cam.wait_ready(timeout_sec=5.0):
        cam.log("[CAM] not ready, stop")
        return

    detector = MarkerBoardDetector()
    cam.log("[WEB] press Prepare to go, then Play Scan")

    robot_state = {
        "prepared": False,
        "motion": None,
        "lock": threading.Lock(),
    }

    while True:
        if cam.consume_prepare_request():
            prepare_robot(cam, robot_state)

        if cam.consume_scan_request():
            result = run_scan_pipeline(cam, detector, board)
            if result.get("ok"):
                perform_robot_move(cam, board, robot_state, detector)

        time.sleep(0.08)


if __name__ == "__main__":
    main()
