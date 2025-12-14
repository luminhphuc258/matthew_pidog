#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tempfile
import subprocess
from pathlib import Path

from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# ===== CHỈNH ĐÚNG THIẾT BỊ CỦA BẠN =====
MIC_DEVICE = "plughw:3,0"      # USB mic (card 3) – hoặc "plughw:4,0" cho BRIO
SPK_DEVICE = "plughw:0,0"      # SunFounder speaker (card 0)
# =====================================

RATE = 16000
CH = 1
FMT = "S16_LE"
REC_SEC = 4
LOOP_DELAY = 0.5


def record_wav(path: str, sec: int) -> bool:
    cmd = [
        "arecord",
        "-D", MIC_DEVICE,
        "-f", FMT,
        "-r", str(RATE),
        "-c", str(CH),
        "-d", str(int(sec)),
        "-q",
        path
    ]
    return subprocess.run(cmd, check=False).returncode == 0


def play_wav(path: str) -> bool:
    # retry nhẹ để tránh busy
    for _ in range(8):
        p = subprocess.run(["aplay", "-D", SPK_DEVICE, "-q", path], check=False)
        if p.returncode == 0:
            return True
        time.sleep(0.2)
    return False


def main():
    # 1) Boot robot => MatthewPidogBootClass.create() sẽ unlock speaker
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    print("=== LOOP RECORD → PLAY (Ctrl+C để dừng) ===")
    print("Mic:", MIC_DEVICE)
    print("Speaker:", SPK_DEVICE)

    while True:
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="mic_")
            os.close(fd)

            print("[REC] Recording 4s...")
            if not record_wav(wav_path, REC_SEC):
                print("[ERROR] arecord failed")
                try: os.unlink(wav_path)
                except: pass
                time.sleep(1)
                continue

            time.sleep(0.2)

            print("[PLAY] Playback...")
            if not play_wav(wav_path):
                print("[ERROR] aplay failed (device busy?)")

            try:
                os.unlink(wav_path)
            except Exception:
                pass

            time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:
            print("\n[EXIT] Stopped.")
            break


if __name__ == "__main__":
    main()
