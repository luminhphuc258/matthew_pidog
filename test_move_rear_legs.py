#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

from motion_controller import MotionController
from move_rear_legs import MoveRearLegs

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"


def set_led_pink(dog):
    try:
        strip = dog.rgb_strip
        strip.set_mode("breath", "pink", bps=0.6)
    except Exception:
        pass


def set_led_off(dog):
    try:
        strip = dog.rgb_strip
        strip.set_mode("off")
    except Exception:
        pass


def main():
    print("[TEST] Boot robot …")
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()
    dog = motion.dog

    print("[TEST] SIT 3s + LED pink")
    set_led_pink(dog)
    motion.sit(speed=20)
    time.sleep(3.0)

    print("[TEST] MoveRearLegs flow …")
    rear = MoveRearLegs(pose_file=POSE_FILE)
    dog = rear.run(dog)

    print("[TEST] STAND")
    set_led_off(dog)
    try:
        dog.do_action("stand", speed=15)
        dog.wait_all_done()
    except Exception:
        pass

    print("[TEST] DONE ✅")


if __name__ == "__main__":
    main()
