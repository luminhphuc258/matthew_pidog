#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

from matthewpidogclassinit import MatthewPidogBootClass
from move_rear_legs import MoveRearLegs

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"


def set_led(dog, color: str, bps: float = 0.8):
    try:
        if dog and hasattr(dog, "rgb_strip"):
            dog.rgb_strip.set_mode("breath", color, bps=bps)
    except Exception:
        pass


def led_off(dog):
    try:
        if dog and hasattr(dog, "rgb_strip"):
            try:
                dog.rgb_strip.set_mode("off")
            except Exception:
                dog.rgb_strip.set_mode("solid", "black")
    except Exception:
        pass


def main():
    # 1) Boot PiDog
    boot = MatthewPidogBootClass()
    dog = boot.create()
    time.sleep(1.0)

    # 2) SIT 3s + LED pink
    print("[TEST] Sit 3s + LED pink...")
    try:
        dog.do_action("sit", speed=20)
        dog.wait_all_done()
    except Exception:
        pass

    set_led(dog, "pink", bps=0.8)
    time.sleep(3.0)

    # 3) Move rear legs (P4/P6) + apply pose config
    print("[TEST] Move rear legs (P4/P6) -> apply pose config...")
    rear = MoveRearLegs(
        pose_file=POSE_FILE,

        # ===== chỉnh theo robot bạn nếu cần =====
        p4_start=80,
        p6_start=-70,
        p4_target=65,
        p6_target=-55,

        delay=0.05,
    )
    rear.run(apply_pose=True)

    # 4) Stand + LED off
    print("[TEST] Stand...")
    try:
        dog.do_action("stand", speed=20)
        dog.wait_all_done()
    except Exception:
        pass

    led_off(dog)
    print("[DONE]")


if __name__ == "__main__":
    main()
