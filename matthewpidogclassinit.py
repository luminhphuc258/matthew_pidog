#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shutil
import subprocess
from time import sleep
from pathlib import Path

from pidog import Pidog
from robot_hat import Servo


class _NoIMU:
    """
    Drop-in replacement for SH3001.
    IMPORTANT: pidog's _imu_thread expects:
        data = imu._sh3001_getimudata()
        accData, gyroData = data
    so we MUST return exactly a 2-tuple: (acc, gyro)
    """
    def __init__(self, *args, **kwargs):
        print("[BOOT] IMU disabled: using _NoIMU (skip SH3001 init)")

    def _sh3001_getimudata(self):
        acc = (0.0, 0.0, 0.0)
        gyro = (0.0, 0.0, 0.0)
        return (acc, gyro)


def _noop_imu_thread(self):
    """No-op imu thread to avoid touching IMU at all."""
    # nếu pidog có loop nền khác thì kệ, ở đây mình cho thread kết thúc luôn
    return


class MatthewPidogBootClass:
    """
    Reusable helper:
    - unlock speaker (SPK_EN)
    - load LEG_INIT_ANGLES from pose file (P0..P7)
    - init Pidog
    - (optional) disable IMU init/read to avoid conflicts with IMU service
    - force head servo after init (bypass pidog)
    - support_stand(): 5-phase slow synchronized servo recovery/stand support
    """

    def __init__(
        self,
        speaker_device: str = "plughw:0,0",
        pose_file: str | Path = "pidog_pose_config.txt",
        leg_pins=None,
        head_pins=None,
        tail_pin=None,
        head_init_angles=None,
        tail_init_angle=None,
        force_head_port: str = "P10",
        force_head_angle: float = -90,
        enable_force_head: bool = True,
        disable_imu: bool = True,   # ✅ NEW
        skip_head_init: bool | None = None,
        skip_mcu_reset: bool | None = None,
    ):
        self.speaker_device = speaker_device
        self.pose_file = Path(pose_file) if not isinstance(pose_file, Path) else pose_file

        self.leg_pins = leg_pins or [0, 1, 2, 3, 4, 5, 6, 7]
        self.head_pins = head_pins or [8, 9, 10]
        self.tail_pin  = tail_pin  or [11]

        # giữ nguyên như code bạn
        self.head_init_angles = head_init_angles or [80, -70, 90]
        self.tail_init_angle  = tail_init_angle or [30]

        self.force_head_port = force_head_port
        self.force_head_angle = force_head_angle
        self.enable_force_head = enable_force_head

        self.disable_imu = disable_imu  # ✅ NEW
        if skip_head_init is None:
            env = os.environ.get("PIDOG_SKIP_HEAD_INIT") or os.environ.get("SKIP_HEAD_INIT")
            self.skip_head_init = str(env).strip().lower() in ("1", "true", "yes", "on")
        else:
            self.skip_head_init = bool(skip_head_init)

        if skip_mcu_reset is None:
            env = os.environ.get("PIDOG_SKIP_MCU_RESET") or os.environ.get("SKIP_MCU_RESET")
            self.skip_mcu_reset = str(env).strip().lower() in ("1", "true", "yes", "on")
        else:
            self.skip_mcu_reset = bool(skip_mcu_reset)

        self.dog: Pidog | None = None

        # clamp range chuẩn robot_hat
        self._angle_min = -90
        self._angle_max = 90

    # ===================== AUDIO UNLOCK (SPK_EN) =====================

    CONFIG_PATHS = ["/boot/firmware/config.txt", "/boot/config.txt"]

    def _read_boot_config(self) -> str:
        for p in self.CONFIG_PATHS:
            if os.path.exists(p):
                try:
                    return open(p, "r", errors="ignore").read()
                except:
                    pass
        return ""

    def detect_spk_en_pin(self) -> int:
        txt = self._read_boot_config()
        lines = [ln.split("#", 1)[0].strip() for ln in txt.splitlines()]
        overlays = [ln for ln in lines if ln.startswith("dtoverlay=")]

        if any("googlevoicehat-soundcard" in ln for ln in overlays):
            return 12
        if any("hifiberry-dac" in ln for ln in overlays):
            return 20

        try:
            out = subprocess.run(["aplay", "-l"], capture_output=True, text=True).stdout.lower()
            if "googlevoi" in out or "googlevoicehat" in out:
                return 12
        except:
            pass

        return 20

    def set_gpio_high(self, pin: int) -> bool:
        if shutil.which("pinctrl"):
            subprocess.run(["pinctrl", "set", str(pin), "op", "dh"], check=False)
            return True
        if shutil.which("raspi-gpio"):
            subprocess.run(["raspi-gpio", "set", str(pin), "op", "dh"], check=False)
            return True
        return False

    def prime_speaker(self):
        import wave
        silence = "/tmp/robothat_silence.wav"
        if not os.path.exists(silence):
            with wave.open(silence, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * (16000 // 10))
        subprocess.run(["aplay", "-D", self.speaker_device, "-q", silence], check=False)

    def unlock_speaker(self) -> bool:
        pin = self.detect_spk_en_pin()
        ok = self.set_gpio_high(pin)
        if not ok:
            print("[WARN] Không tìm thấy pinctrl/raspi-gpio (hoặc thiếu quyền).")
            return False

        self.prime_speaker()
        subprocess.run(["amixer", "sset", "robot-hat speaker", "100%"], check=False)
        subprocess.run(["amixer", "sset", "robot-hat speaker Playback Volume", "100%"], check=False)

        print(f"[OK] Speaker unlocked (SPK_EN GPIO{pin})")
        return True

    # ===================== LOAD LEG_INIT_ANGLES FROM FILE =====================

    def _parse_pose_file(self, path: Path) -> dict[int, float]:
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()

        # JSON first
        try:
            obj = json.loads(txt)
            out = {}
            for k, v in obj.items():
                m = re.search(r"(\d+)", str(k))
                if m:
                    out[int(m.group(1))] = float(v)
            return out
        except Exception:
            pass

        out = {}
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            m = re.search(r"[Pp]\s*(\d+)\s*[:=]\s*(-?\d+(?:\.\d+)?)", line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
                continue

            nums = re.findall(r"-?\d+(?:\.\d+)?", line)
            if len(nums) >= 2:
                ch = int(float(nums[0]))
                ang = float(nums[1])
                out[ch] = ang

        return out

    def load_leg_init_angles(self, fallback=None):
        fallback = fallback or [-3, 89, 9, -80, 3, 90, 10, -90]

        try:
            if not self.pose_file.exists():
                print(f"[WARN] Không thấy {self.pose_file}, dùng fallback LEG_INIT_ANGLES.")
                return fallback

            pose = self._parse_pose_file(self.pose_file)
            missing = [i for i in range(8) if i not in pose]
            if missing:
                print(f"[WARN] Pose file thiếu {missing} (P0..P7). Dùng fallback.")
                return fallback

            leg_angles = [pose[i] for i in range(8)]
            print(f"[OK] Loaded LEG_INIT_ANGLES from {self.pose_file}: {leg_angles}")
            return leg_angles

        except Exception as e:
            print(f"[WARN] Lỗi đọc pose file: {e}. Dùng fallback.")
            return fallback

    def load_head_init_angles(self, fallback=None):
        fallback = fallback or self.head_init_angles or [80, -70, 90]
        try:
            if not self.pose_file.exists():
                return fallback
            pose = self._parse_pose_file(self.pose_file)
            missing = [i for i in (8, 9, 10) if i not in pose]
            if missing:
                return fallback
            return [pose[8], pose[9], pose[10]]
        except Exception:
            return fallback

    # ===================== FORCE SERVO =====================

    def force_servo_angle(self, port: str, angle: float, hold=0.3) -> bool:
        if angle < -90: angle = -90
        if angle > 90:  angle = 90

        try:
            s = Servo(port)
            s.angle(angle)
            sleep(hold)
            print(f"[FORCE] {port} -> {angle}° (bypass pidog)")
            return True
        except Exception as e:
            print(f"[FORCE ERROR] không set được {port}: {e}")
            return False

    # ===================== SUPPORT STAND (NEW) =====================

    def clamp(self, v: int | float) -> int:
        try:
            v = int(v)
        except Exception:
            v = 0
        return max(self._angle_min, min(self._angle_max, v))

    def smooth_pair(
        self,
        pA: str, a_start: int, a_end: int,
        pB: str, b_start: int, b_end: int,
        step: int = 1,
        delay: float = 0.03,
    ):
        sA = Servo(pA)
        sB = Servo(pB)

        a_start, a_end = self.clamp(a_start), self.clamp(a_end)
        b_start, b_end = self.clamp(b_start), self.clamp(b_end)

        a = a_start
        b = b_start

        try:
            sA.angle(a)
            sB.angle(b)
        except Exception:
            pass

        max_steps = max(abs(a_end - a_start), abs(b_end - b_start))
        if max_steps == 0:
            return

        step = max(1, int(abs(step)))

        for _ in range(max_steps):
            if a != a_end:
                a += step if a_end > a else -step
                if (a_end > a_start and a > a_end) or (a_end < a_start and a < a_end):
                    a = a_end

            if b != b_end:
                b += step if b_end > b else -step
                if (b_end > b_start and b > b_end) or (b_end < b_start and b < b_end):
                    b = b_end

            try:
                sA.angle(self.clamp(a))
                sB.angle(self.clamp(b))
            except Exception:
                pass

            sleep(delay)

    def support_stand(self, step: int = 1, delay: float = 0.03, pause_sec: float = 1.0):
        print("[support_stand] Phase 1: P1 +90 -> +5, P3 -76 -> +5")
        self.smooth_pair("P1", +90, +5, "P3", -76, +5, step=step, delay=delay)
        sleep(pause_sec)

        print("[support_stand] Phase 2: P6 +32 -> +63, P4 +4 -> -39")
        self.smooth_pair("P6", +32, +63, "P4", +4, -39, step=step, delay=delay)
        sleep(pause_sec)

        print("[support_stand] Phase 3: P0 -8 -> -55, P2 +15 -> +74")
        self.smooth_pair("P0", -8, -55, "P2", +15, +74, step=step, delay=delay)
        sleep(pause_sec)

        print("[support_stand] Phase 4: P4 -39 -> -2, P6 +63 -> +33")
        self.smooth_pair("P4", -39, -2, "P6", +63, +33, step=step, delay=delay)
        sleep(pause_sec)

        print("[support_stand] Phase 5: P0 -55 -> +12, P2 +74 -> -2")
        self.smooth_pair("P0", -55, +12, "P2", +74, -2, step=step, delay=delay)

        print("[support_stand] DONE")

    # ===================== INIT PIDOG =====================

    def _patch_disable_imu(self):
        """
        Patch pidog internals BEFORE Pidog() init:
        - Replace Sh3001 driver with _NoIMU
        - Replace Pidog._imu_thread with no-op to avoid background exceptions
        """
        if not self.disable_imu:
            return

        try:
            import pidog.pidog as pidog_mod

            # 1) patch IMU driver class
            pidog_mod.Sh3001 = _NoIMU

            # 2) patch imu thread to no-op (prevents any read loop)
            try:
                pidog_mod.Pidog._imu_thread = _noop_imu_thread
            except Exception:
                pass

            print("[BOOT] IMU disabled: patched Sh3001 + patched Pidog._imu_thread (no-op)")
        except Exception as e:
            print(f"[WARN] IMU disable patch failed: {e}")

    def _patch_skip_mcu_reset(self):
        if not self.skip_mcu_reset:
            return
        try:
            import robot_hat.utils as rh_utils
            rh_utils.reset_mcu = lambda: print("[BOOT] MCU reset skipped")
        except Exception:
            pass
        try:
            import pidog.pidog as pidog_mod
            if hasattr(pidog_mod, "utils"):
                pidog_mod.utils.reset_mcu = lambda: print("[BOOT] MCU reset skipped")
        except Exception:
            pass

    def create(self) -> Pidog:
        print("=== PidogBootstrap.create() ===")
        self.unlock_speaker()

        leg_init_angles = self.load_leg_init_angles()

        head_pins = self.head_pins
        head_init_angles = self.head_init_angles
        if self.skip_head_init:
            head_init_angles = self.load_head_init_angles()
            if self.enable_force_head:
                print("[BOOT] skip_head_init=1 -> disable force head")
                self.enable_force_head = False

        print("Init PiDog...")
        print("LEG_PINS :", self.leg_pins,  "angles:", leg_init_angles)
        print("HEAD_PINS:", head_pins, "angles:", head_init_angles)
        print("TAIL_PIN :", self.tail_pin,  "angle :", self.tail_init_angle)

        # ✅ IMPORTANT: disable IMU init/read before creating Pidog
        self._patch_disable_imu()
        self._patch_skip_mcu_reset()

        self.dog = Pidog(
            leg_pins=self.leg_pins,
            head_pins=head_pins,
            tail_pin=self.tail_pin,
            leg_init_angles=leg_init_angles,
            head_init_angles=head_init_angles,
            tail_init_angle=self.tail_init_angle
        )

        if hasattr(self.dog, "wait_all_done"):
            self.dog.wait_all_done()

        if self.enable_force_head:
            sleep(0.2)
            self.force_servo_angle(self.force_head_port, self.force_head_angle, hold=0.4)

        print("[DONE] Pidog created.")
        return self.dog
