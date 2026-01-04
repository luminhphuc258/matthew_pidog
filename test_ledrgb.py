#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pidog.rgb_strip import RGBStrip

def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[ERR] {fn.__name__}: {e}")
        return None

def main():
    strip = RGBStrip()
    print("[OK] RGBStrip() created:", strip)

    # 0) thử show() trước (một số driver cần gọi show để init)
    safe_call(strip.show)
    time.sleep(0.2)

    # 1) solid đỏ / xanh / xanh dương / trắng (dễ nhận)
    for name in ["red", "green", "blue", "white"]:
        print(f"[TEST] Solid {name}")
        safe_call(strip.set_mode, style="solid", color=name, brightness=1.0)
        safe_call(strip.show)
        time.sleep(1.0)

    # 2) chase (nếu support)
    print("[TEST] Chase rainbow (if supported)")
    safe_call(strip.set_mode, style="chase", color="rainbow", bps=6, brightness=1.0)
    safe_call(strip.show)
    time.sleep(3.0)

    # 3) breath xanh dương (như bạn muốn)
    print("[TEST] Breath blue")
    safe_call(strip.set_mode, style="breath", color="blue", bps=1.2, brightness=1.0)
    safe_call(strip.show)
    time.sleep(5.0)

    # 4) tắt
    print("[TEST] OFF")
    safe_call(strip.set_mode, style="off")
    safe_call(strip.show)

    print("[DONE] If you saw nothing at all, likely service/power/pin issue.")

if __name__ == "__main__":
    main()
