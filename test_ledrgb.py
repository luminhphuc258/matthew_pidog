#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

# ====== PATCH: disable SH3001 IMU to avoid crash ======
import pidog.pidog as pidog_mod

class DummyIMU:
    def __init__(self, *args, **kwargs):
        print("[PATCH] DummyIMU enabled: skip SH3001 init")
    def get_acc(self): return (0.0, 0.0, 0.0)
    def get_gyro(self): return (0.0, 0.0, 0.0)
    def get_euler(self): return (0.0, 0.0, 0.0)

pidog_mod.Sh3001 = DummyIMU
# ======================================================

from matthewpidog_boot import MatthewPidogBootClass  # đổi đúng tên file class của bạn

def main():
    boot = MatthewPidogBootClass(enable_force_head=False)
    dog = boot.create()

    strip = dog.rgb_strip
    print("[TEST] breath blue")
    strip.set_mode("breath", "blue", bps=1.2)
    time.sleep(3)

    print("[TEST] breath red")
    strip.set_mode("breath", "red", bps=1.2)
    time.sleep(3)

    print("[TEST] off")
    try:
        strip.set_mode("off", "black")
    except Exception:
        strip.set_mode("breath", "black", bps=1.0)

if __name__ == "__main__":
    main()
