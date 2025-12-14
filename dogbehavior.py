#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class BehaviorOutput:
    decision: str                 # FORWARD | TURN_LEFT | TURN_RIGHT | BACK | STOP
    reason: str
    state: str                    # NORMAL | AVOID_ROTATE | LISTEN_HOLD | REST_SIT
    led_mode: str = "breath"
    led_color: str = "white"
    led_bps: float = 0.4
    face: str = "what_is_it"


class DogBehavior:
    """
    Hành vi robot:
    1) Ưu tiên distance sensor -> obstacle -> xoay 360 độ trong 5s rồi đi tiếp
    2) Khi mic đang thu -> đứng yên (STOP/stand)
    3) Đi 60s -> sit 3s -> stand -> đi tiếp
    4) Camera chỉ dùng khi distance ok hoặc không có distance
    """

    def __init__(
        self,
        safe_dist_cm: float = 50.0,
        emergency_stop_cm: float = 10.0,
        rotate_sec: float = 5.0,
        walk_rest_every_sec: float = 60.0,
        rest_sit_sec: float = 3.0,
        after_avoid_cooldown_sec: float = 1.2,
    ):
        self.safe_dist_cm = float(safe_dist_cm)
        self.emergency_stop_cm = float(emergency_stop_cm)
        self.rotate_sec = float(rotate_sec)
        self.walk_rest_every_sec = float(walk_rest_every_sec)
        self.rest_sit_sec = float(rest_sit_sec)
        self.after_avoid_cooldown_sec = float(after_avoid_cooldown_sec)

        # state machine
        self.state = "NORMAL"
        self.until_ts = 0.0

        # rest timer
        self._walk_start_ts = time.time()

        # cooldown để tránh “xoay xong lại tiến vào đúng chỗ cũ ngay lập tức”
        self._avoid_cooldown_until = 0.0

        # để bắn 1 lần action sit/stand khi vào state
        self._entered_state_ts = 0.0

    def _center_blocked(self, sectors: List[str]) -> int:
        if not sectors:
            return 0
        n = len(sectors)
        c = n // 2
        if n >= 3 and c - 1 >= 0 and c + 1 < n:
            center3 = [sectors[c - 1], sectors[c], sectors[c + 1]]
        else:
            center3 = sectors
        return sum(1 for s in center3 if s == "blocked")

    def reset_walk_timer(self):
        self._walk_start_ts = time.time()

    def should_rest(self) -> bool:
        return (time.time() - self._walk_start_ts) >= self.walk_rest_every_sec

    def start_rest(self):
        self.state = "REST_SIT"
        self.until_ts = time.time() + self.rest_sit_sec
        self._entered_state_ts = time.time()

    def start_avoid_rotate(self):
        self.state = "AVOID_ROTATE"
        self.until_ts = time.time() + self.rotate_sec
        self._entered_state_ts = time.time()

    def start_listen_hold(self):
        self.state = "LISTEN_HOLD"
        self.until_ts = 0.0
        self._entered_state_ts = time.time()

    def set_cooldown(self):
        self._avoid_cooldown_until = time.time() + self.after_avoid_cooldown_sec

    def in_cooldown(self) -> bool:
        return time.time() < self._avoid_cooldown_until

    def update(
        self,
        *,
        dist_cm: Optional[float],
        sector_states: List[str],
        camera_decision: str,
        cam_blocked: bool,
        imu_bump: bool,
        manual_active: bool,
        manual_move: Optional[str],
        is_recording: bool,
        is_playing: bool,
    ) -> BehaviorOutput:
        now = time.time()

        # ========== manual override luôn ưu tiên cao nhất ==========
        if manual_active and manual_move:
            return BehaviorOutput(
                decision=manual_move,
                reason="MANUAL",
                state="MANUAL",
                led_mode="breath",
                led_color="white" if manual_move != "FORWARD" else "blue",
                led_bps=0.6,
                face="what_is_it",
            )

        # ========== behavior nghe chỉ dẫn ==========
        # Khi mic đang thu -> đứng yên. Khi loa phát -> cho chạy tiếp (không hold).
        if is_recording and (not is_playing):
            # giữ state LISTEN_HOLD để web thấy rõ
            self.start_listen_hold()
            return BehaviorOutput(
                decision="STOP",
                reason="LISTENING_MIC_RECORDING",
                state="LISTEN_HOLD",
                led_mode="breath",
                led_color="white",
                led_bps=0.3,
                face="what_is_it",
            )
        else:
            # mic không thu -> cho quay lại NORMAL nếu đang LISTEN_HOLD
            if self.state == "LISTEN_HOLD":
                self.state = "NORMAL"
                self._entered_state_ts = now

        # ========== emergency ==========
        if dist_cm is not None and dist_cm < self.emergency_stop_cm:
            # emergency stop (không xoay)
            return BehaviorOutput(
                decision="STOP",
                reason=f"EMERGENCY_STOP({dist_cm:.1f}cm)",
                state="NORMAL",
                led_mode="breath",
                led_color="red",
                led_bps=0.9,
                face="angry",
            )

        # ========== REST behavior: mỗi 60s sit 3s ==========
        if self.state == "NORMAL" and self.should_rest():
            self.start_rest()

        if self.state == "REST_SIT":
            if now < self.until_ts:
                return BehaviorOutput(
                    decision="SIT",  # main sẽ xử lý SIT bằng do_action trực tiếp
                    reason="REST_SIT_3S",
                    state="REST_SIT",
                    led_mode="breath",
                    led_color="white",
                    led_bps=0.25,
                    face="sleep",
                )
            else:
                # hết sit -> reset walk timer và tiếp tục
                self.state = "NORMAL"
                self._entered_state_ts = now
                self.reset_walk_timer()
                return BehaviorOutput(
                    decision="STOP",
                    reason="REST_DONE_STAND",
                    state="NORMAL",
                    led_mode="breath",
                    led_color="white",
                    led_bps=0.35,
                    face="what_is_it",
                )

        # ========== obstacle detect (ưu tiên SENSOR trước) ==========
        center_blocked = self._center_blocked(sector_states)

        sensor_trigger = (dist_cm is not None and dist_cm < self.safe_dist_cm)

        # camera chỉ dùng sau sensor:
        camera_trigger = (dist_cm is None) and (center_blocked >= 2)

        trigger = sensor_trigger or camera_trigger or imu_bump or cam_blocked

        # ========== avoid rotate state machine ==========
        if self.state == "NORMAL":
            if trigger and (not self.in_cooldown()):
                self.start_avoid_rotate()

        if self.state == "AVOID_ROTATE":
            if now < self.until_ts:
                return BehaviorOutput(
                    decision="TURN_RIGHT",
                    reason="AVOID_ROTATE_360_5S",
                    state="AVOID_ROTATE",
                    led_mode="breath",
                    led_color="red",
                    led_bps=0.8,
                    face="angry",
                )
            else:
                # kết thúc xoay -> cooldown + đi tiếp
                self.state = "NORMAL"
                self._entered_state_ts = now
                self.set_cooldown()
                # reset walk timer vì vừa đổi hướng
                self.reset_walk_timer()
                return BehaviorOutput(
                    decision="STOP",
                    reason="AVOID_DONE_COOLDOWN",
                    state="NORMAL",
                    led_mode="breath",
                    led_color="white",
                    led_bps=0.35,
                    face="what_is_it",
                )

        # ========== NORMAL decision ==========
        # Ưu tiên sensor: nếu dist có và OK -> dùng camera decision
        # Nếu dist không có -> dùng camera decision
        decision = camera_decision

        # LED rule
        if decision == "FORWARD":
            led_color = "blue"
            face = "music"
        elif decision in ("TURN_LEFT", "TURN_RIGHT"):
            led_color = "white"
            face = "what_is_it"
        else:
            led_color = "white"
            face = "what_is_it"

        return BehaviorOutput(
            decision=decision,
            reason="NORMAL",
            state="NORMAL",
            led_mode="breath",
            led_color=led_color,
            led_bps=0.6 if decision != "STOP" else 0.35,
            face=face,
        )
