#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from typing import Dict

from robot_hat import Servo


class MoveRearLegs:
    """
    Phiên bản đảo chiều quay chân sau:
    - P5, P7 đi NGƯỢC chiều so với bản cũ
    - Giữ nguyên lock, timing, pose apply
    """

    def __init__(
        self,
        pose_file: Path,

        # ===== REAR LEGS (P5, P7) =====
        p5_start: int = 18,
        p7_start: int = -13,
        p5_target: int = 4,
        p7_target: int = -1,

        # ===== LOAD LEGS (LOCK) =====
        p4_lock: int = 80,
        p6_lock: int = -70,

        delay: float = 0.05,
        angle_min: int = -90,
        angle_max: int = 90,
    ):
        self.pose_file = Path(pose_file)

        self.P5_START = int(p5_start)
        self.P7_START = int(p7_start)
        self.P5_TARGET = int(p5_target)
        self.P7_TARGET = int(p7_target)

        self.P4_LOCK = int(p4_lock)
        self.P6_LOCK = int(p6_lock)

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

    def apply_pose_from_cfg(
        self,
        cfg: Dict[str, int],
        per_servo_delay: float = 0.03,
        settle_sec: float = 0.8,
    ):
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

        if settle_sec > 0:
            time.sleep(settle_sec)

    # ---------------- core logic (REVERSED) ----------------
    def move_only_rear_legs(self):
        """
        ĐẢO CHIỀU:
        - Bắt đầu từ TARGET
        - Di chuyển về START
        """

        s4 = Servo("P4")
        s5 = Servo("P5")
        s6 = Servo("P6")
        s7 = Servo("P7")

        # STEP 1: lock load legs
        self._apply(s4, self.P4_LOCK)
        self._apply(s6, self.P6_LOCK)
        time.sleep(0.35)

        # 🔁 STEP 2: start từ TARGET (đảo chiều)
        curr_p5 = self.P5_TARGET
        curr_p7 = self.P7_TARGET

        self._apply(s5, curr_p5)
        self._apply(s7, curr_p7)
        self._apply(s4, self.P4_LOCK)
        self._apply(s6, self.P6_LOCK)
        time.sleep(0.45)

        # 🔁 STEP 3: move ngược về START
        while curr_p5 != self.P5_START or curr_p7 != self.P7_START:
            self._apply(s4, self.P4_LOCK)
            self._apply(s6, self.P6_LOCK)

            if curr_p5 != self.P5_START:
                curr_p5 += 1 if self.P5_START > curr_p5 else -1
                self._apply(s5, curr_p5)
                time.sleep(self.DELAY)

            if curr_p7 != self.P7_START:
                curr_p7 += 1 if self.P7_START > curr_p7 else -1
                self._apply(s7, curr_p7)
                time.sleep(self.DELAY)

        time.sleep(0.25)

    # ---------------- public API ----------------
    def run(self):
        """
        Dùng khi robot đang SIT:
        - chỉnh lại chân sau theo chiều ngược
        - apply pose config
        """
        self.move_only_rear_legs()
        cfg = self.load_pose_config()
        self.apply_pose_from_cfg(cfg)
        return True
