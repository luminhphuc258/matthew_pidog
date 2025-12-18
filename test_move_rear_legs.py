#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from typing import Optional

from matthewpidogclassinit import MatthewPidogBootClass


class MoveRearLegs:
    """
    Safe rear-body reset flow using PiDog high-level API

    Flow:
    1) sit
    2) body_stop (robot nằm hẳn)
    3) wait 2s
    4) re-boot sequence (đưa robot về trạng thái chuẩn)
    """

    def __init__(self, pose_file: Path):
        self.pose_file = Path(pose_file)
        self._dog = None

    # ---------- helpers ----------
    def _safe_wait(self, dog):
        try:
            dog.wait_all_done()
        except Exception:
            pass

    def sit(self, dog, speed=20):
        try:
            dog.do_action("sit", speed=speed)
            self._safe_wait(dog)
            return True
        except Exception:
            return False

    def body_stop(self, dog):
        """
        PiDog API: nằm xuống thật sự
        """
        try:
            dog.body_stop()
            self._safe_wait(dog)
            return True
        except Exception:
            return False

    # ---------- main flow ----------
    def run(self, dog) -> Optional[object]:
        """
        Execute rear-leg reset flow.
        Return new dog instance after reboot.
        """
        if dog is None:
            return None

        print("[MoveRearLegs] SIT …")
        self.sit(dog, speed=20)
        time.sleep(0.5)

        print("[MoveRearLegs] BODY STOP (lie down) …")
        self.body_stop(dog)
        time.sleep(2.0)

        print("[MoveRearLegs] REBOOT SEQUENCE …")
        boot = MatthewPidogBootClass()
        new_dog = boot.create()

        time.sleep(1.2)
        return new_dog
