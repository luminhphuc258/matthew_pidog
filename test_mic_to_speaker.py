#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, tempfile, subprocess, re
from pathlib import Path
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

MIC_DEVICE = "plughw:3,0"   # BRIO mic (đổi lại 3,0 nếu dùng USB mic khác)

RATE = 16000
CH = 1
FMT = "S16_LE"
REC_SEC = 4
LOOP_DELAY = 0.5


def detect_voicehat_playback_device() -> str:
    out = subprocess.run(["aplay", "-l"], capture_output=True, text=True).stdout.lower()

    # match "card N: ... voicehat/googlevoicehat ..."
    m = re.search(r"card\s+(\d+):[^\n]*(googlevoicehat|voicehat)", out)
    if m:
        return f"plughw:{m.group(1)},0"

    # fallback by driver id in brackets
    m2 = re.search(r"card\s+(\d+):[^\n]*\[\s*snd_rpi_googlevoicehat_soundcard\s*\]", out)
    if m2:
        return f"plughw:{m2.group(1)},0"

    return "default"


def record_wav(path: str, sec: int) -> bool:
    cmd = ["arecord","-D",MIC_DEVICE,"-f",FMT,"-r",str(RATE),"-c",str(CH),"-d",str(int(sec)),"-q",path]
    return subprocess.run(cmd, check=False).returncode == 0

def free_snd_devices():
    # tìm PID đang giữ /dev/snd/*
    p = subprocess.run(["sudo", "fuser", "-v", "/dev/snd/*"],
                       text=True, capture_output=True)
    txt = (p.stdout or "") + (p.stderr or "")
    pids = sorted(set(re.findall(r"\b(\d+)\b", txt)), key=int)

    # không có ai giữ
    if not pids:
        return

    print("[AUDIO] busy, killing:", " ".join(pids))
    for pid in pids:
        subprocess.run(["sudo", "kill", "-9", pid], check=False)
    time.sleep(0.2)

def play_wav(device: str, path: str) -> bool:
    free_snd_devices()  # ✅ giải phóng trước khi play

    for _ in range(12):
        p = subprocess.run(["aplay", "-D", device, "-q", path], check=False)
        if p.returncode == 0:
            return True
        time.sleep(0.2)
    return False



def main():
    # boot robot (unlock speaker)
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    spk_device = detect_voicehat_playback_device()
    print("=== LOOP RECORD → PLAY (Ctrl+C để dừng) ===")
    print("Mic:", MIC_DEVICE)
    print("Speaker auto:", spk_device)

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
            if not play_wav(spk_device, wav_path):
                print("[ERROR] aplay failed (busy or wrong device)")

            try: os.unlink(wav_path)
            except: pass

            time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:
            print("\n[EXIT]")
            break


if __name__ == "__main__":
    main()
