#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLOW theo yêu cầu:
1) boot bình thường (pre-move + apply pose_config + create dog + stand)
2) stand
3) sit
4) chờ 2s
5) apply "body_stop_pose" (đúng vị trí servo bạn đưa) bằng Servo(Px).angle(...)
6) chờ settle
7) reboot sequence lại: boot -> stand
"""

import time
from pathlib import Path
from motion_controller import MotionController
from robot_hat import Servo

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# Pose "body stop" bạn gửi
BODY_STOP_POSE = {
    "P0": 12,
    "P1": 5,
    "P2": -2,
    "P3": 5,
    "P4": 15,
    "P5": 25,
    "P6": 11,
    "P7": -15,
    "P8": 39,
    "P9": -70,
    "P10": 84,
    "P11": 0,
}


def clamp(v, lo=-90, hi=90):
    try:
        v = int(v)
    except Exception:
        v = 0
    return max(lo, min(hi, v))


def apply_pose_direct(pose: dict, per_servo_delay=0.03, settle_sec=1.0):
    """
    Apply pose trực tiếp bằng robot_hat.Servo("Px").angle(...)
    Không phụ thuộc dog.do_action, giúp robot nằm đúng "body stop" pose.
    """
    servos = {}
    for p in [f"P{i}" for i in range(12)]:
        try:
            servos[p] = Servo(p)
        except Exception:
            pass

    for p, ang in pose.items():
        s = servos.get(p)
        if s is None:
            continue
        try:
            s.angle(clamp(ang))
        except Exception:
            pass
        time.sleep(per_servo_delay)

    if settle_sec and settle_sec > 0:
        time.sleep(settle_sec)


def main():
    print("=== TEST MOVE REAR (FLOW + BODY_STOP_POSE) ===")

    mc = MotionController(pose_file=POSE_FILE)

    # ===== ROUND 1 =====
    print("[1] BOOT (includes apply pose_config + create dog + stand)")
    mc.boot()
    time.sleep(0.8)

    print("[2] STAND (confirm)")
    mc.stand(speed=30, force=True)
    time.sleep(0.8)

    print("[3] SIT")
    mc.sit(speed=20)

    print("[4] WAIT 2s on SIT")
    time.sleep(2.0)

    print("[5] APPLY BODY_STOP_POSE (direct servo angles)")
    apply_pose_direct(BODY_STOP_POSE, per_servo_delay=0.03, settle_sec=1.2)

    # ===== ROUND 2 =====
    print("[6] REBOOT SEQUENCE AGAIN: BOOT -> STAND")
    mc.boot()
    time.sleep(0.8)

    mc.stand(speed=30, force=True)
    time.sleep(0.5)

    print("[DONE] Completed.")


if __name__ == "__main__":
    main()
