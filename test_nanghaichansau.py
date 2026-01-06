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


def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    os.environ.setdefault("PIDOG_SKIP_HEAD_INIT", "1")
    os.environ.setdefault("PIDOG_SKIP_MCU_RESET", "1")

    motion = MotionController(pose_file=POSE_FILE)

    try:
        motion.boot()

        print("[TEST] lift rear legs")
        apply_angles(REAR_LIFT_ANGLES, per_servo_delay=0.04)
        print("[TEST] rear lift done")
        time.sleep(3.0)

        print("[TEST] lift front legs")
        apply_angles(FRONT_LIFT_ANGLES, per_servo_delay=0.04)
        print("[TEST] front lift done")

        print("[TEST] done")

    finally:
        try:
            motion.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
