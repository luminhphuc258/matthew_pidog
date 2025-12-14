#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import wave
import shutil
import tempfile
import subprocess
from pathlib import Path

from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# ====== chỉnh 2 cái này đúng với máy bạn ======
MIC_DEVICE = "plughw:4,0"        # USB microphone
SPEAKER_DEVICE = "plughw:0,0"    # Robot HAT / SunFounder speaker card
# ============================================

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = "S16_LE"


def _set_gpio_high(pin: int) -> bool:
    """Bật SPK_EN lên HIGH (cần sudo)."""
    if shutil.which("pinctrl"):
        subprocess.run(["pinctrl", "set", str(pin), "op", "dh"], check=False)
        return True
    if shutil.which("raspi-gpio"):
        subprocess.run(["raspi-gpio", "set", str(pin), "op", "dh"], check=False)
        return True
    return False


def _prime_speaker(device: str):
    """Phát 0.1s silence để 'mở' đường audio."""
    silence = "/tmp/robothat_silence.wav"
    if not os.path.exists(silence):
        with wave.open(silence, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * (16000 // 10))  # 0.1s
    subprocess.run(["aplay", "-D", device, "-q", silence], check=False)


def unlock_sunfounder_speaker(device: str) -> bool:
    """
    Thử mở khóa speaker theo 2 pin phổ biến:
      - GPIO12 (googlevoicehat-soundcard)
      - GPIO20 (hifiberry-dac)
    """
    for pin in (12, 20):
        _set_gpio_high(pin)
        _prime_speaker(device)
        # test: nếu aplay prime chạy xong coi như ok
        # (không có cách "confirm" tuyệt đối, nhưng thực tế đủ dùng)
        print(f"[SPK] Tried unlock with GPIO{pin}")
        return True

    print("[SPK] Cannot unlock (missing pinctrl/raspi-gpio?)")
    return False


def record_wav(path: str, seconds: int):
    cmd = [
        "arecord",
        "-D", MIC_DEVICE,
        "-f", FORMAT,
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-d", str(int(seconds)),
        "-q",
        path
    ]
    subprocess.run(cmd, check=False)


def play_wav(path: str):
    cmd = ["aplay", "-D", SPEAKER_DEVICE, "-q", path]
    subprocess.run(cmd, check=False)


def main():
    # 1) boot motion (đúng cách bạn yêu cầu)
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    # 2) unlock speaker trước
    unlock_sunfounder_speaker(SPEAKER_DEVICE)

    # 3) record 4s rồi phát lại
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="mic_")
    os.close(fd)

    print("[TEST] Recording 4s from USB mic...")
    record_wav(wav_path, 4)

    time.sleep(0.2)

    print("[TEST] Playing back to SunFounder speaker...")
    play_wav(wav_path)

    try:
        os.unlink(wav_path)
    except Exception:
        pass

    print("[DONE] Mic loopback finished.")


if __name__ == "__main__":
    main()
