#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path

from robot_hat import Servo
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# Boot lift angles (same flow as test_nanghaichansau.py)
REAR_LIFT_ANGLES = {
    "P4": 80,   # rear hip left
    "P5": 30,   # rear knee left
    "P6": -70,  # rear hip right
    "P7": -30,  # rear knee right
}

FRONT_LIFT_ANGLES = {
    "P0": -20,  # front hip left
    "P1": 90,   # front knee left
    "P2": 20,   # front hip right
    "P3": -75,  # front knee right
}

HEAD_INIT_ANGLES = {
    "P8": 80,   # head yaw
    "P9": -70,  # head roll
    "P10": 90,  # head pitch
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


def smooth_single(
    port: str,
    start: int,
    end: int,
    step: int = 1,
    delay: float = 0.03,
):
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


def ha_hai_chan_truoc():
    print("[PHASE] ha chan truoc: P0->+28, P1->+6, P2->-13, P3->0")
    smooth_single("P0", 0, 28, step=1, delay=0.03)
    smooth_single("P1", 0, 6, step=1, delay=0.03)
    smooth_single("P2", 0, -13, step=1, delay=0.03)
    smooth_single("P3", 0, 0, step=1, delay=0.03)
    print("[PHASE] ha chan truoc done")


def ha_hai_chan_sau():
    print("[PHASE] ha chan sau: P7->+19, P6->-24, P4->+46, P5->-24")
    smooth_single("P7", 0, 19, step=1, delay=0.03)
    smooth_single("P6", 0, -24, step=1, delay=0.03)
    smooth_single("P4", 0, 46, step=1, delay=0.03)
    smooth_single("P5", 0, -24, step=1, delay=0.03)
    print("[PHASE] ha chan sau done")


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

    print("[PHASE] draw done -> ha chan truoc + ha chan sau")
    ha_hai_chan_truoc()
    ha_hai_chan_sau()
    print("[PHASE] robotvehinhcaro done")


def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    os.environ.setdefault("PIDOG_SKIP_HEAD_INIT", "1")
    os.environ.setdefault("PIDOG_SKIP_MCU_RESET", "1")

    print("[BOOT] set head init angles")
    apply_angles(HEAD_INIT_ANGLES, per_servo_delay=0.04)
    print("[BOOT] head init done (hold position)")

    print("[BOOT] lift rear legs (left then right)")
    smooth_pair("P4", 0, REAR_LIFT_ANGLES["P4"], "P5", 0, REAR_LIFT_ANGLES["P5"], step=1, delay=0.03)
    smooth_pair("P6", 0, REAR_LIFT_ANGLES["P6"], "P7", 0, REAR_LIFT_ANGLES["P7"], step=1, delay=0.03)
    print("[BOOT] rear lift done (hold position)")
    time.sleep(2.0)

    print("[BOOT] lift front legs")
    apply_angles(FRONT_LIFT_ANGLES, per_servo_delay=0.04)
    print("[BOOT] front lift done (hold position)")
    time.sleep(2.0)

    print("[BOOT] boot robot to stand")
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()
    print("[BOOT] boot done")

    print("[RUN] robotvehinhcaro")
    robotvehinhcaro()
    print("[DONE]")


if __name__ == "__main__":
    main()
