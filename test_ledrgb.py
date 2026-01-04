#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import traceback

from pidog.rgb_strip import RGBStrip
from matthewpidogclassinit import MatthewPidogBootClass

def safe_call(name, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[ERR] {name}: {e}")
        traceback.print_exc()
        return None


def test_led(strip: RGBStrip):
    print("[TEST] Begin LED test...")

    # 0) init/show
    safe_call("strip.show()", strip.show)
    time.sleep(0.2)

    # 1) Solid colors
    for c in ["red", "green", "blue", "white"]:
        print(f"[TEST] solid {c}")
        safe_call("strip.set_mode(solid)", strip.set_mode, style="solid", color=c, brightness=1.0)
        safe_call("strip.show()", strip.show)
        time.sleep(1.0)

    # 2) Chase (nếu lib support)
    print("[TEST] chase rainbow (if supported)")
    safe_call("strip.set_mode(chase)", strip.set_mode, style="chase", color="rainbow", bps=6, brightness=1.0)
    safe_call("strip.show()", strip.show)
    time.sleep(3.0)

    # 3) Breath (như bạn dùng)
    print("[TEST] breath blue")
    safe_call("strip.set_mode(breath)", strip.set_mode, style="breath", color="blue", bps=1.2, brightness=1.0)
    safe_call("strip.show()", strip.show)
    time.sleep(5.0)

    # 4) OFF
    print("[TEST] off")
    safe_call("strip.set_mode(off)", strip.set_mode, style="off")
    safe_call("strip.show()", strip.show)

    print("[DONE] LED test finished.")


def main():
    print("=== Boot MatthewPidog first (to unlock board/power) ===")
    boot = MatthewPidogBootClass(
        pose_file="pidog_pose_config.txt",   # nếu file nằm chỗ khác thì sửa path
        enable_force_head=False              # test LED thì khỏi force head cho nhẹ
    )

    # Quan trọng: create() để init robot_hat / pidog
    dog = safe_call("boot.create()", boot.create)
    if dog is None:
        print("[FATAL] boot.create() failed -> LED likely won't work.")
        return

    # (Tuỳ chọn) đứng yên để tránh servo rung
    if hasattr(dog, "stop"):
        safe_call("dog.stop()", dog.stop)

    print("=== Now init RGBStrip and test ===")
    strip = RGBStrip()
    test_led(strip)


if __name__ == "__main__":
    main()
