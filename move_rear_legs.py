#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from typing import Dict
from robot_hat import Servo


class MoveRearLegs:
    """
    Nâng 2 chân sau bằng 2 motor P4 và P6 (từ từ, đều, không giật).
    - Start -> Target theo từng 1 degree
    - Alternating (P4 rồi P6) để robot ổn định.
    - Cuối cùng apply pose config (tất cả servo) nếu bạn muốn.
    """

    def __init__(
        self,
        pose_file: Path,

        # ===== REAR LIFT (P4, P6) =====
        p4_start: int = 80,
        p6_start: int = -70,
        p4_target: int = 65,   # bạn chỉnh theo thực tế (nâng lên)
        p6_target: int = -55,  # bạn chỉnh theo thực tế (nâng lên)

        delay: float = 0.05,
        angle_min: int = -90,
        angle_max: int = 90,
    ):
        self.pose_file = Path(pose_file)

        self.P4_START = int(p4_start)
        self.P6_START = int(p6_start)
        self.P4_TARGET = int(p4_target)
        self.P6_TARGET = int(p6_target)

        self.DELAY = float(delay)
        self.angle_min = int(angle_min)
        self.angle_max = int(angle_max)

        self.servo_ports = [f"P{i}" for i in range(12)]

    # ---------------- utils ----------------
    def clamp(self, v: int) -> int:
        try:
            v = int(v)
        except Exception:
            v = 0
        return max(self.angle_min, min(self.angle_max, v))

    def _apply(self, servo: Servo, angle: int):
        try:
            servo.angle(self.clamp(angle))
        except Exception:
            pass

    # ---------------- pose ----------------
    def load_pose_config(self) -> Dict[str, int]:
        cfg = {p: 0 for p in self.servo_ports}
        if not self.pose_file.exists():
            return cfg
        try:
            data = json.loads(self.pose_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in cfg:
                        cfg[k] = self.clamp(v)
        except Exception:
            pass
        return cfg

    def apply_pose_from_cfg(self, cfg: Dict[str, int], per_servo_delay: float = 0.03, settle_sec: float = 0.8):
        servos = {}
        for p in self.servo_ports:
            try:
                servos[p] = Servo(p)
            except Exception:
                pass

        for p in self.servo_ports:
            s = servos.get(p)
            if not s:
                continue
            self._apply(s, cfg.get(p, 0))
            time.sleep(per_servo_delay)

        if settle_sec and settle_sec > 0:
            time.sleep(settle_sec)

    # ---------------- core: lift rear legs using P4/P6 ----------------
    def lift_rear_legs_smooth(self):
        s4 = Servo("P4")
        s6 = Servo("P6")

        # 1) set start (nhẹ nhàng)
        curr4 = self.P4_START
        curr6 = self.P6_START
        self._apply(s4, curr4)
        self._apply(s6, curr6)
        time.sleep(0.6)

        # 2) alternating step-by-step
        while curr4 != self.P4_TARGET or curr6 != self.P6_TARGET:
            if curr4 != self.P4_TARGET:
                curr4 += 1 if self.P4_TARGET > curr4 else -1
                self._apply(s4, curr4)
                time.sleep(self.DELAY)

            if curr6 != self.P6_TARGET:
                curr6 += 1 if self.P6_TARGET > curr6 else -1
                self._apply(s6, curr6)
                time.sleep(self.DELAY)

        time.sleep(0.25)

    def run(self, apply_pose: bool = True):
        """
        - Nâng 2 chân sau (P4/P6) từ từ
        - Option: apply pose config sau đó
        """
        self.lift_rear_legs_smooth()
        if apply_pose:
            cfg = self.load_pose_config()
            self.apply_pose_from_cfg(cfg)
        return True
