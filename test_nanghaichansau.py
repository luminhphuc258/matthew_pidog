#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path

from robot_hat import Servo
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# Tune these angles for your robot.
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


def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    os.environ.setdefault("PIDOG_SKIP_HEAD_INIT", "1")
    os.environ.setdefault("PIDOG_SKIP_MCU_RESET", "1")

    print("[TEST] set head init angles")
    apply_angles(HEAD_INIT_ANGLES, per_servo_delay=0.04)
    print("[TEST] head init done (hold position)")

    print("[TEST] lift front legs")
    apply_angles(FRONT_LIFT_ANGLES, per_servo_delay=0.04)
    print("[TEST] front lift done (hold position)")

    print("[TEST] lift rear legs (smooth)")
    smooth_pair("P4", 0, REAR_LIFT_ANGLES["P4"], "P6", 0, REAR_LIFT_ANGLES["P6"], step=1, delay=0.03)
    smooth_pair("P5", 0, REAR_LIFT_ANGLES["P5"], "P7", 0, REAR_LIFT_ANGLES["P7"], step=1, delay=0.03)
    print("[TEST] rear lift done (hold position)")

    print("[TEST] done")


if __name__ == "__main__":
    main()
