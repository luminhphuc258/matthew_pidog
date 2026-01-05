#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, time, random, threading
from pathlib import Path
from typing import Optional

from robot_hat import Servo
from matthewpidogclassinit import MatthewPidogBootClass
from pidog.preset_actions import bark, push_up


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
        head_p8_idle=65,        # idle/stop angle for P8
        head_p9_fixed=-68,
        head_p10_a=90,
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
        self.head_p8_idle = head_p8_idle
        self.head_p9_fixed = head_p9_fixed
        self.head_p10_a = head_p10_a
        self.head_p10_b = head_p10_b

        self._dog = None

        # head thread control
        self._head_stop_evt: Optional[threading.Event] = None
        self._head_thread: Optional[threading.Thread] = None
        self._head_lock = threading.Lock()
        self._head_mode = "IDLE"   # IDLE / MOVE
        self._s8 = None
        self._s9 = None
        self._s10 = None

        self._lock = threading.Lock()

        # tránh spam stand/stop liên tục
        self._last_stand_ts = 0.0
        self._stand_cooldown = 0.25

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

    # ---------------- boot sequence ----------------
    def _pre_move_legs_before_pose(self):
        """
        1) lock P4/P6
        2) move rear P5/P7 alternating
        3) set front hip P0/P2
        4) move front knee P1/P3 alternating (P1 invert)
        """
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

    # =========================
    # HEAD CONTROLLER (P8 sweep when moving)
    # =========================
    def _set_head_mode(self, mode: str):
        with self._head_lock:
            self._head_mode = str(mode or "IDLE").upper()

    def start_head_controller(self):
        """
        P8: sweep smoothly between 10..60 deg ONLY when moving (FORWARD / TURN_LEFT / TURN_RIGHT)
        P9: fixed
        P10: toggle a/b as old behavior (optional)
        """
        stop_evt = threading.Event()

        try:
            self._s8 = Servo("P8")
            self._s9 = Servo("P9")
            self._s10 = Servo("P10")
        except Exception:
            self._s8 = self._s9 = self._s10 = None
            return stop_evt, None

        # configurable by ENV
        sweep_min = float(__import__("os").environ.get("HEAD_SWEEP_MIN", "70.0"))
        sweep_max = float(__import__("os").environ.get("HEAD_SWEEP_MAX", "70.0"))
        speed_dps = float(__import__("os").environ.get("HEAD_SWEEP_SPEED_DPS", "55.0"))  # deg/sec
        tick_sec  = float(__import__("os").environ.get("HEAD_TICK_SEC", "0.03"))         # 30ms
        idle_p8   = float(__import__("os").environ.get("HEAD_P8_IDLE", str(self.head_p8_idle)))

        # clamp sweep range
        sweep_min = float(self.clamp(sweep_min))
        sweep_max = float(self.clamp(sweep_max))
        if sweep_max < sweep_min:
            sweep_min, sweep_max = sweep_max, sweep_min

        # init position near idle
        pos = float(self.clamp(idle_p8))
        direction = +1.0

        # p10 toggle like old
        target_p10 = self.head_p10_b
        next_flip = time.time() + random.uniform(0.6, 1.6)

        def worker():
            nonlocal pos, direction, target_p10, next_flip
            while not stop_evt.is_set():
                # read mode
                with self._head_lock:
                    mode = self._head_mode

                # always keep P9 fixed
                try:
                    if self._s9:
                        self._s9.angle(self.clamp(self.head_p9_fixed))
                except Exception:
                    pass

                # P10 optional toggle (kept as your old)
                try:
                    now = time.time()
                    if now >= next_flip:
                        target_p10 = self.head_p10_a if target_p10 == self.head_p10_b else self.head_p10_b
                        next_flip = now + random.uniform(0.6, 1.6)
                    if self._s10:
                        self._s10.angle(self.clamp(target_p10))
                except Exception:
                    pass

                # P8 behavior
                try:
                    if self._s8:
                        if mode == "MOVE":
                            # smooth sweep: step = speed * dt
                            step = max(0.2, speed_dps * tick_sec)  # 최소 step nhỏ để không bị đứng
                            pos += direction * step
                            if pos >= sweep_max:
                                pos = sweep_max
                                direction = -1.0
                            elif pos <= sweep_min:
                                pos = sweep_min
                                direction = +1.0
                            self._s8.angle(self.clamp(pos))
                        else:
                            # idle: gently return to idle angle (no jerk)
                            step = max(0.2, speed_dps * tick_sec)
                            if abs(pos - idle_p8) <= step:
                                pos = idle_p8
                            else:
                                pos += step if (idle_p8 > pos) else -step
                            self._s8.angle(self.clamp(pos))
                except Exception:
                    pass

                time.sleep(max(0.01, tick_sec))

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return stop_evt, t

    def boot(self):
        """
        A) pre-move 4 legs (rear+front) with lock P4/P6
        B) apply pose from cfg (all servos)
        C) boot MatthewPidogBootClass -> stand
        D) start head controller (P8 sweep on MOVE)
        """
        self._pre_move_legs_before_pose()
        skip_pose = str(__import__("os").environ.get("SKIP_APPLY_POSE", "0")).lower() in ("1", "true", "yes", "on")
        if not skip_pose:
            cfg = self.load_pose_config()
            self.apply_pose_from_cfg(cfg, per_servo_delay=0.03, settle_sec=1.0)

        boot = MatthewPidogBootClass(skip_head_init=True, enable_force_head=False)
        self._dog = boot.create()
        time.sleep(1.0)
        self.stand(speed=30)

        # optional bark once
        try:
            bark(self._dog, [0, 0, -40])
            time.sleep(0.2)
        except Exception:
            pass

        # start head controller thread
        self._set_head_mode("IDLE")
        self._head_stop_evt, self._head_thread = self.start_head_controller()

    def close(self):
        if self._head_stop_evt is not None:
            self._head_stop_evt.set()
        if self._head_thread is not None:
            self._head_thread.join(timeout=0.5)

    # ---------------- high-level actions ----------------
    def stand(self, speed=10, force=False):
        """Stand nhẹ, có cooldown để khỏi spam."""
        if self._dog is None:
            return
        now = time.time()
        if (not force) and (now - self._last_stand_ts < self._stand_cooldown):
            return
        self._last_stand_ts = now
        try:
            self._dog.do_action("stand", speed=speed)
            self._dog.wait_all_done()
        except Exception:
            pass

    def sit(self, speed=20):
        if self._dog is None:
            return
        try:
            self._dog.do_action("sit", speed=speed)
            self._dog.wait_all_done()
            return
        except Exception:
            pass
        self.stand(speed=10)

    def do_push_up(self):
        if self._dog is None:
            return
        try:
            push_up(self._dog)
        except Exception:
            self.stand(speed=10)

    def do_bark(self):
        if self._dog is None:
            return
        try:
            bark(self._dog, [0, 0, -40])
        except Exception:
            pass

    # ---------------- decision executor ----------------
    def execute(self, decision: str):
        """
        decision (string):
          - Auto: FORWARD | TURN_LEFT | TURN_RIGHT | BACK | STOP
          - Manual buttons: LEFT | RIGHT
          - Idle actions: STAND | SIT | PUSH_UP | BARK
        """
        if self._dog is None:
            return

        d = (decision or "STOP").upper().strip()
        # alias
        if d == "LEFT":
            d = "TURN_LEFT"
        if d == "RIGHT":
            d = "TURN_RIGHT"

        moving_cmds = {"FORWARD", "TURN_LEFT", "TURN_RIGHT"}

        with self._lock:
            try:
                # ---- set head mode before movement ----
                if d in moving_cmds:
                    self._set_head_mode("MOVE")
                else:
                    self._set_head_mode("IDLE")

                if d == "FORWARD":
                    self._dog.do_action("forward", speed=250)
                    self._dog.wait_all_done()

                elif d == "BACK":
                    self._dog.do_action("backward", speed=250)
                    self._dog.wait_all_done()

                elif d == "TURN_LEFT":
                    self._dog.do_action("turn_left", step_count=1, speed=230)
                    self._dog.wait_all_done()

                elif d == "TURN_RIGHT":
                    self._dog.do_action("turn_right", step_count=1, speed=230)
                    self._dog.wait_all_done()

                elif d == "SIT":
                    self._set_head_mode("IDLE")
                    self.sit(speed=20)

                elif d == "STAND":
                    self._set_head_mode("IDLE")
                    self.stand(speed=10, force=True)

                elif d == "PUSH_UP":
                    self._set_head_mode("IDLE")
                    self.do_push_up()

                elif d == "BARK":
                    self._set_head_mode("IDLE")
                    self.do_bark()

                else:
                    # STOP: chỉ giữ stand nhẹ thôi, không spam
                    self._set_head_mode("IDLE")
                    self.stand(speed=5, force=False)

            except Exception:
                pass
            finally:
                # after movement ends, return head to idle smoothly
                if d in moving_cmds:
                    self._set_head_mode("IDLE")
