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
music.music_set_volume(80)  # tuỳ bạn

print("▶️ Playing tiengsua.wav ...")
music.music_play(str(wav), loop=1)  # phát 1 lần (đúng theo guide)

# ✅ GIỮ CHƯƠNG TRÌNH SỐNG để nhạc phát hết
# Cách 1: sleep đủ dài (ví dụ 10s)
time.sleep(10)

# Hoặc Cách 2 (khuyên dùng): chờ bạn Ctrl+C mới dừng
# print("Press Ctrl+C to stop...")
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     pass

print("⏹ stop")
music.music_stop()
