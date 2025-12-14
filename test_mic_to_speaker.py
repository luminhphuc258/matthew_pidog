#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, tempfile, subprocess
from pathlib import Path
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

MIC_DEVICE = "plughw:3,0"  # hoặc "plughw:4,0" nếu dùng BRIO
SPK_DEVICE = "plughw:snd_rpi_googlevoicehat_soundcard,0"  # KHÔNG ĐỔI SAU REBOOT

RATE = 16000
CH = 1
FMT = "S16_LE"
REC_SEC = 4
LOOP_DELAY = 0.5


def record_wav(path: str, sec: int) -> bool:
    cmd = ["arecord","-D",MIC_DEVICE,"-f",FMT,"-r",str(RATE),"-c",str(CH),"-d",str(int(sec)),"-q",path]
    return subprocess.run(cmd, check=False).returncode == 0


def play_wav(path: str) -> bool:
    for _ in range(12):
        p = subprocess.run(["aplay", "-D", SPK_DEVICE, "-q", path], check=False)
        if p.returncode == 0:
            return True
        time.sleep(0.2)
    return False


def main():
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()  # MatthewPidogBootClass đã unlock speaker

    print("Mic:", MIC_DEVICE)
    print("Speaker:", SPK_DEVICE)
    print("=== LOOP RECORD → PLAY (Ctrl+C để dừng) ===")

    while True:
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="mic_")
            os.close(fd)

            print("[REC] 4s...")
            if not record_wav(wav_path, REC_SEC):
                print("[ERROR] arecord failed")
                try: os.unlink(wav_path)
                except: pass
                time.sleep(1)
                continue

            time.sleep(0.15)

            print("[PLAY] ...")
            if not play_wav(wav_path):
                print("[ERROR] aplay failed (device busy or wrong device)")

            try: os.unlink(wav_path)
            except: pass

            time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:
            print("\n[EXIT]")
            break


if __name__ == "__main__":
    main()
