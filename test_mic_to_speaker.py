#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tempfile
import subprocess
from pathlib import Path

from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# ======= THIẾT BỊ CỦA BẠN (đúng theo aplay -l / arecord -l) =======
MIC_DEVICE = "plughw:3,0"      # BRIO mic (card 4). Nếu dùng USB mic khác: "plughw:3,0"
SPK_HW     = "plughw:1,0"      # Google VoiceHAT SoundCard (card 1)
# ===================================================================

RATE = 16000
CH = 1
FMT = "S16_LE"
REC_SEC = 4
LOOP_DELAY = 0.5

# Nếu hw bị busy, sẽ thử dùng "default" (dmix) nếu bạn cấu hình /etc/asound.conf
SPK_FALLBACK = "default"

# -------------------------------------------------------------

def run(cmd, capture=False):
    if capture:
        return subprocess.run(cmd, text=True, capture_output=True)
    return subprocess.run(cmd, check=False)

def list_holders_of_snd():
    """Trả về text fuser -v /dev/snd/*"""
    p = run(["sudo", "fuser", "-v", "/dev/snd/*"], capture=True)
    return (p.stdout or "") + (p.stderr or "")

def kill_common_audio_daemons():
    """
    Tắt các thứ hay chiếm audio (an toàn trong môi trường robot).
    pipewire/pulse bạn đã tắt rồi, nhưng giữ lại để chắc.
    """
    run(["pkill", "-9", "pipewire"], capture=False)
    run(["pkill", "-9", "pipewire-pulse"], capture=False)
    run(["pkill", "-9", "pulseaudio"], capture=False)

def kill_processes_holding_snd():
    """
    Kill các process đang giữ /dev/snd/* (cứng tay nhưng hiệu quả).
    Dùng fuser để lấy PID.
    """
    txt = list_holders_of_snd()
    pids = []
    for token in txt.split():
        if token.isdigit():
            pids.append(token)

    # lọc trùng
    pids = sorted(set(pids), key=lambda x: int(x))

    if not pids:
        return False

    print("[AUDIO] /dev/snd/* is busy. Killing PIDs:", ", ".join(pids))
    for pid in pids:
        run(["sudo", "kill", "-9", pid], capture=False)

    time.sleep(0.2)
    return True

def ensure_audio_free():
    """
    Cố gắng giải phóng audio trước khi play.
    """
    kill_common_audio_daemons()
    kill_processes_holding_snd()

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

def try_aplay(device: str, wav_path: str, tries=12, wait=0.2) -> bool:
    for _ in range(tries):
        p = subprocess.run(["aplay", "-D", device, "-q", wav_path], check=False)
        if p.returncode == 0:
            return True
        time.sleep(wait)
    return False

def play_wav(wav_path: str) -> bool:
    # 1) giải phóng audio trước khi play
    ensure_audio_free()

    # 2) thử play trực tiếp hw trước
    if try_aplay(SPK_HW, wav_path, tries=10, wait=0.2):
        return True

    # 3) nếu vẫn busy, thử fallback "default" (dmix) nếu bạn có cấu hình
    print("[WARN] aplay hw busy -> trying fallback:", SPK_FALLBACK)
    if try_aplay(SPK_FALLBACK, wav_path, tries=10, wait=0.2):
        return True

    # 4) nếu vẫn fail, in ra ai đang giữ
    print("[ERROR] aplay failed. Current holders:\n", list_holders_of_snd())
    return False

# -------------------------------------------------------------

def main():
    # 1) Boot robot (unlock speaker + init pidog)
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    print("=== LOOP RECORD → PLAY (Ctrl+C để dừng) ===")
    print("Mic:", MIC_DEVICE)
    print("Speaker(HW):", SPK_HW, " fallback:", SPK_FALLBACK)
    print()

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

            time.sleep(0.15)

            print("[PLAY] Playback...")
            ok = play_wav(wav_path)
            if not ok:
                print("[ERROR] Playback failed (busy).")

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
