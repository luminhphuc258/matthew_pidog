#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tempfile
import subprocess
from pathlib import Path

from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# ✅ dùng default theo /etc/asound.conf của bạn
MIC_DEVICE = "default"
SPK_DEVICE = "default"

RATE = 16000
CH = 1
FMT = "S16_LE"
REC_SEC = 4
LOOP_DELAY = 0.3

# tăng gain software nếu bạn thấy nhỏ (1.0 = không đổi, 2.0 = +6dB, 4.0 = +12dB)
SOFT_GAIN = 2.0


def run(cmd):
    return subprocess.run(cmd, check=False)


def set_volumes():
    """
    Cố gắng set volume cho robot-hat speaker/mic nếu control tồn tại.
    Không crash nếu không có.
    """
    # Speaker volume
    run(["amixer", "-q", "sset", "robot-hat speaker", "100%"])
    run(["amixer", "-q", "sset", "robot-hat speaker Playback Volume", "100%"])

    # Mic volume (nếu có)
    run(["amixer", "-q", "sset", "robot-hat mic", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic Capture Volume", "100%"])


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
    return run(cmd).returncode == 0


def boost_wav_inplace(wav_path: str, gain: float) -> bool:
    """
    Tăng âm lượng file WAV bằng sox (nếu có) hoặc ffmpeg.
    Ưu tiên sox vì nhanh và sạch.
    """
    if gain <= 1.01:
        return True

    # thử sox
    if subprocess.run(["bash", "-lc", "command -v sox >/dev/null 2>&1"]).returncode == 0:
        tmp = wav_path + ".tmp.wav"
        # -G để tránh clip mạnh, vol <gain>
        p = run(["sox", wav_path, tmp, "vol", str(gain)])
        if p.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, wav_path)
            return True
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except:
            pass

    # fallback ffmpeg
    if subprocess.run(["bash", "-lc", "command -v ffmpeg >/dev/null 2>&1"]).returncode == 0:
        tmp = wav_path + ".tmp.wav"
        p = run(["ffmpeg", "-y", "-i", wav_path, "-filter:a", f"volume={gain}", tmp])
        if p.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, wav_path)
            return True
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except:
            pass

    return False


def play_wav(path: str) -> bool:
    cmd = ["aplay", "-D", SPK_DEVICE, "-q", path]
    return run(cmd).returncode == 0


def main():
    # 1) boot robot (MatthewPidogBootClass.create() unlock speaker)
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    # 2) set volume
    set_volumes()

    print("=== LOOP RECORD → PLAY (Ctrl+C để dừng) ===")
    print("Mic:", MIC_DEVICE)
    print("Speaker:", SPK_DEVICE)
    print("REC_SEC:", REC_SEC, "SOFT_GAIN:", SOFT_GAIN)
    print()

    while True:
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="mic_")
            os.close(fd)

            print("[REC] 4s...")
            if not record_wav(wav_path, REC_SEC):
                print("[ERROR] arecord failed")
                try:
                    os.unlink(wav_path)
                except:
                    pass
                time.sleep(0.6)
                continue

            # tăng âm lượng software nếu cần
            boost_wav_inplace(wav_path, SOFT_GAIN)

            time.sleep(0.1)

            print("[PLAY] ...")
            if not play_wav(wav_path):
                print("[ERROR] aplay failed")

            try:
                os.unlink(wav_path)
            except:
                pass

            time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:
            print("\n[EXIT] Stopped.")
            break


if __name__ == "__main__":
    main()
