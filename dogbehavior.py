#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BehaviorOutput:
    decision: str          # FORWARD | TURN_LEFT | TURN_RIGHT | BACK | STOP | SIT
    reason: str
    state: str             # NORMAL | MIC_HOLD | BACKING | ROTATING | COOLDOWN | REST_SIT
    led_mode: str = "breath"
    led_color: str = "white"
    led_bps: float = 0.4
    face: str = "what_is_it"


class DogBehavior:
    """
    Fix 3 vấn đề bạn gặp:
    1) Sensor UART ra chậm -> dùng 'freshness' để biết có tin được không.
       - sensor_fresh -> ưu tiên sensor
       - sensor_stale -> fallback camera (cẩn thận hơn)
    2) Khi gặp vật cản: BACK ngắn -> ROTATE 360 (5s) -> đi tiếp
       => tránh kiểu lùi/tiến ngay tại chỗ cũ.
    3) Mic đang record -> robot đứng yên.
    4) Sau 60s di chuyển -> SIT 3s.
    """

    def __init__(
        self,
        safe_dist_cm: float = 50.0,
        emergency_stop_cm: float = 10.0,
        sensor_fresh_sec: float = 0.35,     # ✅ quan trọng: quá thời gian này coi như stale
        back_sec: float = 0.7,              # BACK để thoát vùng cũ
        rotate_sec: float = 5.0,            # 360 ước lượng theo thời gian
        cooldown_sec: float = 1.2,          # sau xoay xong, đừng lao ngay
        walk_rest_every_sec: float = 60.0,
        rest_sit_sec: float = 3.0,
        camera_trigger_center_blocked: int = 2,  # center 3 ô bị blocked >=2 -> coi là cản
    ):
        self.safe_dist_cm = float(safe_dist_cm)
        self.emergency_stop_cm = float(emergency_stop_cm)
        self.sensor_fresh_sec = float(sensor_fresh_sec)

        self.back_sec = float(back_sec)
        self.rotate_sec = float(rotate_sec)
        self.cooldown_sec = float(cooldown_sec)

        self.walk_rest_every_sec = float(walk_rest_every_sec)
        self.rest_sit_sec = float(rest_sit_sec)

        self.camera_trigger_center_blocked = int(camera_trigger_center_blocked)

        self.state = "NORMAL"
        self.until_ts = 0.0

        self._walk_start_ts = time.time()
        self._cooldown_until = 0.0

    # ---------- helpers ----------
    def _center_blocked(self, sectors: List[str]) -> int:
        if not sectors:
            return 0
        n = len(sectors)
        c = n // 2
        if n >= 3 and 0 <= c - 1 and c + 1 < n:
            center3 = [sectors[c - 1], sectors[c], sectors[c + 1]]
        else:
            center3 = sectors
        return sum(1 for s in center3 if s == "blocked")

    def _total_blocked(self, sectors: List[str]) -> int:
        return sum(1 for s in sectors if s == "blocked") if sectors else 0

    def _moving_decision(self, d: str) -> bool:
        return d in ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "BACK")

    def _reset_walk_timer(self):
        self._walk_start_ts = time.time()

    def _should_rest(self) -> bool:
        return (time.time() - self._walk_start_ts) >= self.walk_rest_every_sec

    def _in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    # ---------- main update ----------
    def update(
        self,
        *,
        dist_cm: Optional[float],
        dist_age_sec: Optional[float],     # ✅ tuổi của sample UART
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

        # ====== MANUAL ưu tiên cao nhất ======
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

        # ====== MIC HOLD: mic đang record thì đứng yên ======
        # (kể cả is_playing True/False, cứ record là HOLD)
        if is_recording:
            self.state = "MIC_HOLD"
            return BehaviorOutput(
                decision="STOP",
                reason="MIC_RECORDING_HOLD",
                state="MIC_HOLD",
                led_mode="breath",
                led_color="white",
                led_bps=0.25,
                face="what_is_it",
            )
        else:
            if self.state == "MIC_HOLD":
                self.state = "NORMAL"
                self._reset_walk_timer()

        # ====== REST: mỗi 60s SIT 3s ======
        if self.state == "NORMAL" and self._should_rest():
            self.state = "REST_SIT"
            self.until_ts = now + self.rest_sit_sec

        if self.state == "REST_SIT":
            if now < self.until_ts:
                return BehaviorOutput(
                    decision="SIT",
                    reason="REST_SIT",
                    state="REST_SIT",
                    led_mode="breath",
                    led_color="white",
                    led_bps=0.22,
                    face="sleep",
                )
            # hết sit -> stand/stop 1 nhịp rồi đi tiếp
            self.state = "NORMAL"
            self._reset_walk_timer()
            return BehaviorOutput(
                decision="STOP",
                reason="REST_DONE",
                state="NORMAL",
                led_mode="breath",
                led_color="white",
                led_bps=0.30,
                face="what_is_it",
            )

        # ====== sensor freshness ======
        sensor_fresh = False
        if dist_cm is not None and dist_age_sec is not None:
            sensor_fresh = (dist_age_sec <= self.sensor_fresh_sec)

        center_blk = self._center_blocked(sector_states)
        total_blk = self._total_blocked(sector_states)

        # ====== TRIGGER logic (sensor ưu tiên nếu fresh) ======
        trigger_sensor = False
        trigger_emergency = False
        if sensor_fresh and dist_cm is not None:
            trigger_emergency = dist_cm < self.emergency_stop_cm
            trigger_sensor = dist_cm < self.safe_dist_cm

        # fallback camera nếu sensor stale hoặc không có
        trigger_camera = False
        if (not sensor_fresh) or (dist_cm is None):
            trigger_camera = (center_blk >= self.camera_trigger_center_blocked) or cam_blocked

        trigger = trigger_emergency or trigger_sensor or trigger_camera or imu_bump

        # ====== tránh “đụng rồi mới né”: nếu sensor stale nhưng camera thấy blocked nhiều -> né sớm ======
        if (not sensor_fresh) and (center_blk >= self.camera_trigger_center_blocked):
            trigger = True

        # ====== STATE MACHINE AVOID ======
        # flow: BACK (0.7s) -> ROTATE (5s) -> COOLDOWN (1.2s) -> NORMAL
        if self.state == "NORMAL":
            if (not self._in_cooldown()) and trigger:
                # vào BACKING trước (để thoát chỗ cũ)
                self.state = "BACKING"
                self.until_ts = now + self.back_sec

        if self.state == "BACKING":
            if now < self.until_ts:
                return BehaviorOutput(
                    decision="BACK",
                    reason=f"AVOID_BACK({self.back_sec:.1f}s) dist={dist_cm} age={dist_age_sec}",
                    state="BACKING",
                    led_mode="breath",
                    led_color="red",
                    led_bps=0.9,
                    face="angry",
                )
            # hết BACK -> vào ROTATE
            self.state = "ROTATING"
            self.until_ts = now + self.rotate_sec

        if self.state == "ROTATING":
            if now < self.until_ts:
                return BehaviorOutput(
                    decision="TURN_RIGHT",
                    reason=f"AVOID_ROTATE_360({self.rotate_sec:.1f}s) blkC={center_blk} blkT={total_blk}",
                    state="ROTATING",
                    led_mode="breath",
                    led_color="red",
                    led_bps=0.8,
                    face="angry",
                )
            # hết ROTATE -> cooldown rồi NORMAL
            self.state = "COOLDOWN"
            self.until_ts = now + self.cooldown_sec
            self._cooldown_until = self.until_ts
            self._reset_walk_timer()

        if self.state == "COOLDOWN":
            if now < self.until_ts:
                return BehaviorOutput(
                    decision="STOP",
                    reason=f"COOLDOWN({self.cooldown_sec:.1f}s)",
                    state="COOLDOWN",
                    led_mode="breath",
                    led_color="white",
                    led_bps=0.30,
                    face="what_is_it",
                )
            self.state = "NORMAL"

        # ====== NORMAL decision ======
        decision = camera_decision

        # nếu sensor fresh và quá gần -> đừng FORWARD (an toàn)
        if sensor_fresh and dist_cm is not None and dist_cm < self.safe_dist_cm:
            decision = "STOP"

        # LED + face mapping
        if decision == "FORWARD":
            led_color, face = "blue", "music"
        elif decision in ("TURN_LEFT", "TURN_RIGHT"):
            led_color, face = "white", "what_is_it"
        elif decision == "BACK":
            led_color, face = "red", "angry"
        else:
            led_color, face = "white", "what_is_it"

        return BehaviorOutput(
            decision=decision,
            reason=f"NORMAL sensor_fresh={sensor_fresh} dist={dist_cm} age={dist_age_sec}",
            state="NORMAL",
            led_mode="breath",
            led_color=led_color,
            led_bps=0.6 if self._moving_decision(decision) else 0.35,
            face=face,
        )
