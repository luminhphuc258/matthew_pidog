#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from matthewpidogclassinit import MatthewPidogBootClass 

def main():
    boot = MatthewPidogBootClass(
        pose_file="pidog_pose_config.txt",
        enable_force_head=False
    )
    dog = boot.create()

    if not hasattr(dog, "rgb_strip") or dog.rgb_strip is None:
        print("[FATAL] dog.rgb_strip not found -> lib/firmware issue")
        return

    strip = dog.rgb_strip
    print("[OK] Using dog.rgb_strip:", type(strip))

    # Test 1: breath blue (như bạn hay dùng)
    print("[TEST] breath blue")
    strip.set_mode("breath", "white", bps=1.2)
    time.sleep(3)

    # Test 2: đổi màu để chắc chắn LED nhận lệnh
    print("[TEST] breath red")
    strip.set_mode("breath", "red", bps=1.2)
    time.sleep(3)

    # Test 3: off (nếu supported)
    print("[TEST] off")
    try:
        strip.set_mode("off", "black")
    except Exception:
        # vài version dùng "black" hoặc "none"
        try:
            strip.set_mode("breath", "black", bps=1.0)
        except Exception as e:
            print("[WARN] cannot turn off cleanly:", e)

    print("[DONE]")

if __name__ == "__main__":
    main()
