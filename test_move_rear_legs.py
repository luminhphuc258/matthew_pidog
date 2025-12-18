#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLOW mới:
1) boot bình thường
2) apply pose config
3) stand
4) sit
5) body_stop (nằm thực sự)
6) chờ 2s
7) làm lại: boot -> apply pose -> stand
"""

import time
from pathlib import Path
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"


def main():
    mc = MotionController()

    # ===== ROUND 1 =====
    print("[1] BOOT")
    mc.boot()
    time.sleep(1.2)

    print("[2] APPLY POSE CONFIG:", POSE_FILE)
    mc.apply_pose_config(str(POSE_FILE))
    time.sleep(0.8)

    print("[3] STAND")
    mc.stand()
    time.sleep(1.0)

    print("[4] SIT")
    mc.sit()
    time.sleep(1.0)

    print("[5] BODY_STOP (lie down)")
    mc.body_stop()
    time.sleep(2.0)

    # ===== ROUND 2 (reboot sequence again) =====
    print("[6] REBOOT SEQUENCE AGAIN: BOOT -> APPLY POSE -> STAND")
    mc.boot()
    time.sleep(1.2)

    mc.apply_pose_config(str(POSE_FILE))
    time.sleep(0.8)

    mc.stand()
    time.sleep(1.0)

    print("[DONE] Flow completed. Robot is standing.")


if __name__ == "__main__":
    main()
