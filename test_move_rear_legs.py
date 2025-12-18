#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from robot_hat import Servo

from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"


def clamp(v, lo=-90, hi=90):
    try:
        v = int(v)
    except Exception:
        v = 0
    return max(lo, min(hi, v))


def smooth_pair(pA: str, a_start: int, a_end: int,
                pB: str, b_start: int, b_end: int,
                step=1, delay=0.03):
    """
    Di chuyển 2 servo đồng bộ, mượt:
    - Mỗi tick: cập nhật cả A và B rồi sleep(delay)
    - Không giật vì thay đổi nhỏ theo step
    """
    sA = Servo(pA)
    sB = Servo(pB)

    a_start, a_end = clamp(a_start), clamp(a_end)
    b_start, b_end = clamp(b_start), clamp(b_end)

    a = a_start
    b = b_start

    # set initial
    try:
        sA.angle(a)
        sB.angle(b)
    except Exception:
        pass

    # số bước tối đa theo độ lệch lớn hơn
    max_steps = max(abs(a_end - a_start), abs(b_end - b_start))
    if max_steps == 0:
        return

    for i in range(max_steps):
        # tiến 1 bước về phía target cho từng servo (nếu chưa tới)
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


def support_stand(step=1, delay=0.03):
    """
    SUPPORT STAND - 5 phases (đúng yêu cầu)
    Mỗi phase: 1 cặp motor chạy đồng bộ + chậm, xong nghỉ 1s.
    """

    print("[support_stand] Phase 1: P1 +90 -> +5, P3 -76 -> +5")
    smooth_pair("P1", +90, +5, "P3", -76, +5, step=step, delay=delay)
    time.sleep(1.0)

    print("[support_stand] Phase 2: P6 +32 -> +63, P4 +4 -> -39")
    smooth_pair("P6", +32, +63, "P4", +4, -39, step=step, delay=delay)
    time.sleep(1.0)

    print("[support_stand] Phase 3: P0 -8 -> -55, P2 +15 -> +74")
    smooth_pair("P0", -8, -55, "P2", +15, +74, step=step, delay=delay)
    time.sleep(1.0)

    print("[support_stand] Phase 4: P4 -39 -> -2, P6 +63 -> +33")
    smooth_pair("P4", -39, -2, "P6", +63, +33, step=step, delay=delay)
    time.sleep(1.0)

    print("[support_stand] Phase 5: P0 -55 -> +12, P2 +74 -> -2")
    smooth_pair("P0", -55, +12, "P2", +74, -2, step=step, delay=delay)
    time.sleep(0.2)

    print("[support_stand] DONE")


def main():
    print("=== FLOW: boot -> stand -> sit -> support_stand -> wait 1s -> boot again -> stand ===")

    mc = MotionController(pose_file=POSE_FILE)

    print("[1] BOOT")
    mc.boot()
    time.sleep(0.8)

    print("[2] STAND")
    mc.stand(speed=30, force=True)
    time.sleep(0.8)

    print("[3] SIT")
    mc.sit(speed=20)
    time.sleep(0.8)

    print("[4] CALL support_stand()")
    support_stand(step=1, delay=0.03)

    print("[5] WAIT 1s")
    time.sleep(1.0)

    print("[6] BOOT AGAIN (reboot sequence)")
    mc.boot()
    time.sleep(0.8)

    print("[7] STAND")
    mc.stand(speed=30, force=True)
    time.sleep(0.5)

    print("[DONE]")


if __name__ == "__main__":
    main()
