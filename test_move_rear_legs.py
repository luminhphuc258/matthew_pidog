#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from typing import Optional, Dict

from robot_hat import Servo


class MoveRearLegs:
    """
    Safe "soft reboot" flow (NO new PiDog instance → tránh lỗi MCRST busy)

    Flow:
      1) sit
      2) body_stop (nằm hẳn)
      3) wait 2s
      4) apply pose config again (Servo raw, smooth)
      5) stand (reuse existing dog)
    """

    def __init__(
        self,
        pose_file: Path,
        per_servo_delay: float = 0.03,
        settle_sec: float = 0.8,
    ):
        self.pose_file = Path(pose_file)
        self.per_servo_delay = float(per_servo_delay)
        self.settle_sec = float(settle_sec)

        self.servo_ports = [f"P{i}" for i in range(12)]
        self.angle_min, self.angle_max = -90, 90

    # ---------- helpers ----------
    def _clamp(self, v: float) -> int:
        try:
            v = int(v)
        except Exception:
            v = 0
        return max(self.angle_min, min(self.angle_max, v))

    def _safe_wait(self, dog):
        try:
            dog.wait_all_done()
        except Exception:
            pass

    def _sit(self, dog, speed=20):
        try:
            dog.do_action("sit", speed=speed)
            self._safe_wait(dog)
            return True
        except Exception:
            return False

    def _body_stop(self, dog):
        try:
            dog.body_stop()
            self._safe_wait(dog)
            return True
        except Exception:
            return False

    def _stand(self, dog, speed=15):
        try:
            dog.do_action("stand", speed=speed)
            self._safe_wait(dog)
            return True
        except Exception:
            return False

    def load_pose_config(self) -> Dict[str, int]:
        cfg = {p: 0 for p in self.servo_ports}
        if not self.pose_file.exists():
            return cfg
        try:
            data = json.loads(self.pose_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in cfg:
                        cfg[k] = self._clamp(v)
        except Exception:
            pass
        return cfg

    def apply_pose_from_cfg(self, cfg: Dict[str, int]):
        servos = {}
        for p in self.servo_ports:
            try:
                servos[p] = Servo(p)
            except Exception:
                pass

        for p in self.servo_ports:
            s = servos.get(p)
            if s is None:
                continue
            try:
                s.angle(self._clamp(cfg.get(p, 0)))
                time.sleep(self.per_servo_delay)
            except Exception:
                pass

        if self.settle_sec > 0:
            time.sleep(self.settle_sec)

    # ---------- main flow ----------
    def run(self, dog) -> bool:
        if dog is None:
            return False

        print("[MoveRearLegs] SIT …")
        self._sit(dog, speed=20)
        time.sleep(0.4)

        print("[MoveRearLegs] BODY STOP (lie down) …")
        self._body_stop(dog)
        time.sleep(2.0)

        print("[MoveRearLegs] APPLY POSE CONFIG (soft reboot) …")
        cfg = self.load_pose_config()
        self.apply_pose_from_cfg(cfg)

        print("[MoveRearLegs] STAND …")
        self._stand(dog, speed=15)

        return True
