#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLOW mới (đúng với MotionController hiện tại):
1) boot bình thường (trong boot đã: pre-move -> load pose -> apply pose -> create dog -> stand)
2) stand (gọi lại cho chắc)
3) sit
4) body_stop
5) chờ 2s
6) làm lại "reboot sequence": boot -> stand
"""

import time
from pathlib import Path
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"


def main():
    print("=== TEST FLOW: BOOT → STAND → SIT → BODY_STOP → BOOT AGAIN ===")

    # MotionController bắt buộc pose_file
    mc = MotionController(pose_file=POSE_FILE)

    # ===== ROUND 1 =====
    print("[1] BOOT (includes pose apply + create dog + stand)")
    mc.boot()
    time.sleep(0.8)

    print("[2] STAND (confirm)")
    mc.stand(speed=30, force=True)
    time.sleep(0.8)

    print("[3] SIT")
    mc.sit(speed=20)
    time.sleep(0.8)

    # body_stop là hàm của dog/lib, MotionController chưa wrap -> gọi trực tiếp qua mc.dog
    print("[4] BODY_STOP (robot nằm thật sự)")
    if mc.dog is not None:
        try:
            mc.dog.body_stop()
        except Exception as e:
            print("!! body_stop failed:", e)

    time.sleep(2.0)

    # ===== ROUND 2 =====
    print("[5] BOOT AGAIN (reboot sequence in software)")
    mc.boot()
    time.sleep(0.8)

    print("[6] STAND (confirm)")
    mc.stand(speed=30, force=True)
    time.sleep(0.5)

    print("[DONE] Flow completed.")


if __name__ == "__main__":
    main()
