#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

from motion_controller import MotionController
from move_rear_legs import MoveRearLegs

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"


def set_led(dog, mode: str):
    """
    mode: "pink" | "off"
    """
    if dog is None:
        return
    try:
        strip = getattr(dog, "rgb_strip", None)
        if strip is None:
            return
        if mode == "pink":
            strip.set_mode("breath", "pink", bps=0.7)
        else:
            try:
                strip.set_mode("off")
            except Exception:
                strip.set_mode("solid", "black")
    except Exception:
        pass


def main():
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    dog = getattr(motion, "dog", None)

    print("[TEST] Sit 3s + LED pink...")
    set_led(dog, "pink")
    motion.sit(speed=20)
    time.sleep(3.0)

    print("[TEST] Move rear legs -> apply pose config...")
    rear = MoveRearLegs(
        pose_file=POSE_FILE,
        # dùng y hệt defaults MotionController của bạn (đang đúng)
        p5_start=18, p7_start=-13,
        p5_target=4, p7_target=-1,
        p4_lock=80, p6_lock=-70,
        delay=0.05,
    )
    rear.run()

    print("[TEST] Stand + LED off...")
    set_led(dog, "off")
    motion.stand(speed=30, force=True)

    print("[DONE]")

    # giữ 1 chút để bạn nhìn
    time.sleep(2.0)
    motion.close()


if __name__ == "__main__":
    main()
