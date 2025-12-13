#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, time, random, threading
from pathlib import Path
from typing import Optional

from robot_hat import Servo
from matthewpidogclassinit import MatthewPidogBootClass
from pidog.preset_actions import bark


class MotionController:
    def __init__(
        self,
        pose_file: Path,
        # ===== REAR LEGS (P5, P7) =====
        p5_start=18,
        p7_start=-13,
        p5_target=4,
        p7_target=-1,
        # ===== LOAD LEGS (LOCK) =====
        p4_lock=80,
        p6_lock=-70,
        # ===== FRONT HIP (P0, P2) =====
        p0_target=40,
        p2_target=-26,
        # ===== FRONT KNEE (P1, P3) =====
        p1_start=-25,
        p1_target=-65,
        p3_start=4,
        p3_target=-68,
        p1_invert=True,
        # ===== head controller =====
        head_p8_fixed=30,
        head_p9_fixed=-80,
        head_p10_a=88,
        head_p10_b=90,
    ):
        self.pose_file = Path(pose_file)
        self.servo_ports = [f"P{i}" for i in range(12)]
        self.angle_min, self.angle_max = -90, 90

        # leg pre-move params
        self.P5_START = p5_start
        self.P7_START = p7_start
        self.P5_TARGET = p5_target
        self.P7_TARGET = p7_target

        self.P4_LOCK = p4_lock
        self.P6_LOCK = p6_lock

        self.P0_TARGET = p0_target
        self.P2_TARGET = p2_target

        self.P1_START = p1_start
        self.P1_TARGET = p1_target
        self.P3_START = p3_start
        self.P3_TARGET = p3_target

        self.P1_INVERT = bool(p1_invert)

        self.DELAY = 0.05

        # head controller params
        self.head_p8_fixed = head_p8_fixed
        self.head_p9_fixed = head_p9_fixed
        self.head_p10_a = head_p10_a
        self.head_p10_b = head_p10_b

        self._dog = None
        self._head_stop_evt: Optional[threading.Event] = None
        self._head_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def dog(self):
        return self._dog

    # ---------------- utils ----------------
    def clamp(self, v):
        try:
            v = int(v)
        except Exception:
            v = 0
        return max(self.angle_min, min(self.angle_max, v))

    def apply_angle(self, servo, angle):
        try:
            servo.angle(self.clamp(angle))
        except Exception:
            pass

    def apply_angle_p1(self, servo, angle):
        a = -angle if self.P1_INVERT else angle
        self.apply_angle(servo, a)

    def load_pose_config(self) -> dict:
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

    def apply_pose_from_cfg(self, cfg: dict, per_servo_delay=0.03, settle_sec=1.0):
        servos = {}
        for p in self.servo_ports:
            try:
                servos[p] = Servo(p)
            except Exception:
                pass

        for p in self.servo_ports:
            if p not in servos:
                continue
            try:
                self.apply_angle(servos[p], cfg.get(p, 0))
                time.sleep(per_servo_delay)
            except Exception:
                pass

        if settle_sec and settle_sec > 0:
            time.sleep(settle_sec)

    # ---------------- boot sequence (NEW) ----------------
    def _pre_move_legs_before_pose(self):
        """
        Làm đúng y như script bạn đưa:
        1) init P0..P7
        2) lock P4/P6
        3) move rear P5/P7 alternating
        4) set front hip P0/P2
        5) move front knee P1/P3 alternating (P1 invert)
        """
        # init servos P0..P7
        s0 = Servo("P0")
        s1 = Servo("P1")
        s2 = Servo("P2")
        s3 = Servo("P3")
        s4 = Servo("P4")
        s5 = Servo("P5")
        s6 = Servo("P6")
        s7 = Servo("P7")

        # STEP 1: lock P4,P6
        self.apply_angle(s4, self.P4_LOCK)
        self.apply_angle(s6, self.P6_LOCK)
        time.sleep(0.5)

        # STEP 2: set P5,P7 start
        curr_P5 = self.P5_START
        curr_P7 = self.P7_START
        self.apply_angle(s5, curr_P5)
        self.apply_angle(s7, curr_P7)
        self.apply_angle(s4, self.P4_LOCK)
        self.apply_angle(s6, self.P6_LOCK)
        time.sleep(1.0)

        # STEP 3: alternating move rear P5,P7
        while curr_P5 != self.P5_TARGET or curr_P7 != self.P7_TARGET:
            self.apply_angle(s4, self.P4_LOCK)
            self.apply_angle(s6, self.P6_LOCK)

            if curr_P5 != self.P5_TARGET:
                curr_P5 += 1 if self.P5_TARGET > curr_P5 else -1
                self.apply_angle(s5, curr_P5)
                self.apply_angle(s4, self.P4_LOCK)
                self.apply_angle(s6, self.P6_LOCK)
                time.sleep(self.DELAY)

            if curr_P7 != self.P7_TARGET:
                curr_P7 += 1 if self.P7_TARGET > curr_P7 else -1
                self.apply_angle(s7, curr_P7)
                self.apply_angle(s4, self.P4_LOCK)
                self.apply_angle(s6, self.P6_LOCK)
                time.sleep(self.DELAY)

        # STEP 4: set P0,P2
        self.apply_angle(s0, self.P0_TARGET)
        self.apply_angle(s2, self.P2_TARGET)
        self.apply_angle(s4, self.P4_LOCK)
        self.apply_angle(s6, self.P6_LOCK)
        time.sleep(1.0)

        # STEP 5: move P1,P3 alternating
        curr_P1 = self.P1_START
        curr_P3 = self.P3_START
        self.apply_angle_p1(s1, curr_P1)
        self.apply_angle(s3, curr_P3)

        while curr_P1 != self.P1_TARGET or curr_P3 != self.P3_TARGET:
            self.apply_angle(s4, self.P4_LOCK)
            self.apply_angle(s6, self.P6_LOCK)

            if curr_P1 != self.P1_TARGET:
                curr_P1 += 1 if self.P1_TARGET > curr_P1 else -1
                self.apply_angle_p1(s1, curr_P1)
                self.apply_angle(s4, self.P4_LOCK)
                self.apply_angle(s6, self.P6_LOCK)
                time.sleep(self.DELAY)

            if curr_P3 != self.P3_TARGET:
                curr_P3 += 1 if self.P3_TARGET > curr_P3 else -1
                self.apply_angle(s3, curr_P3)
                self.apply_angle(s4, self.P4_LOCK)
                self.apply_angle(s6, self.P6_LOCK)
                time.sleep(self.DELAY)

        time.sleep(0.3)

    def start_head_controller(self, write_interval=0.08, hold_range=(0.6, 1.6)):
        stop_evt = threading.Event()
        try:
            s8 = Servo("P8")
            s9 = Servo("P9")
            s10 = Servo("P10")
        except Exception:
            return stop_evt, None

        def worker():
            target = self.head_p10_b
            next_flip = time.time() + random.uniform(*hold_range)
            while not stop_evt.is_set():
                now = time.time()
                if now >= next_flip:
                    target = self.head_p10_a if target == self.head_p10_b else self.head_p10_b
                    next_flip = now + random.uniform(*hold_range)
                try:
                    s8.angle(self.clamp(self.head_p8_fixed))
                    s9.angle(self.clamp(self.head_p9_fixed))
                    s10.angle(self.clamp(target))
                except Exception:
                    pass
                time.sleep(write_interval)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return stop_evt, t

    def boot(self):
        """
        BOOT FLOW (NEW):
        A) pre-move 4 legs (rear+front) with lock P4/P6
        B) apply pose from cfg (all servos)
        C) boot MatthewPidogBootClass -> stand
        D) start head controller
        """
        # A) pre-move legs
        self._pre_move_legs_before_pose()

        # B) apply cfg pose (all servos)
        cfg = self.load_pose_config()
        self.apply_pose_from_cfg(cfg, per_servo_delay=0.03, settle_sec=1.0)

        # C) boot dog + stand
        boot = MatthewPidogBootClass()
        self._dog = boot.create()
        time.sleep(1.0)
        try:
            self._dog.do_action("stand", speed=30)
            self._dog.wait_all_done()
            time.sleep(0.5)
        except Exception:
            pass

        # optional bark
        try:
            bark(self._dog, [0, 0, -40])
            time.sleep(0.2)
        except Exception:
            pass

        # D) head lock/wiggle
        self._head_stop_evt, self._head_thread = self.start_head_controller()

    def close(self):
        if self._head_stop_evt is not None:
            self._head_stop_evt.set()
        if self._head_thread is not None:
            self._head_thread.join(timeout=0.5)

    def execute(self, decision: str):
        """
        decision: FORWARD | TURN_LEFT | TURN_RIGHT | BACK | STOP
        """
        if self._dog is None:
            return
        with self._lock:
            try:
                if decision == "FORWARD":
                    self._dog.do_action("forward", speed=250)
                    self._dog.wait_all_done()
                elif decision == "BACK":
                    self._dog.do_action("backward", speed=250)
                    self._dog.wait_all_done()
                elif decision == "TURN_LEFT":
                    self._dog.do_action("turn_left", step_count=1, speed=230)
                    self._dog.wait_all_done()
                elif decision == "TURN_RIGHT":
                    self._dog.do_action("turn_right", step_count=1, speed=230)
                    self._dog.wait_all_done()
                else:
                    self._dog.do_action("stand", speed=5)
                    self._dog.wait_all_done()
            except Exception:
                pass
