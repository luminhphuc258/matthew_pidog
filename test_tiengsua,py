#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from robot_hat import Music

# unlock speaker
os.system("pinctrl set 12 op dh")
time.sleep(0.2)

wav = Path(__file__).resolve().parent / "tiengsua.wav"
if not wav.exists():
    print("❌ Missing file:", wav)
    raise SystemExit(1)

music = Music()
music.music_set_volume(80)

print("▶️ Playing tiengsua.wav (1 time)")
music.music_play(str(wav), loops=1)

# giữ chương trình sống để nhạc phát hết
print("⏳ Waiting for audio to finish...")
time.sleep(8)   # chỉnh theo độ dài file wav

print("⏹ stop")
music.music_stop()
