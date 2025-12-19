#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import inspect
from pathlib import Path

# unlock speaker (SPK_EN = GPIO12)
print("[TEST] Unlock speaker GPIO12")
os.system("pinctrl set 12 op dh")
time.sleep(0.3)

from robot_hat import Music

wav = Path(__file__).resolve().parent / "tiengsua.wav"
if not wav.exists():
    print("[ERROR] tiengsua.wav NOT FOUND:", wav)
    exit(1)

music = Music()

print("\n[TEST] Music.music_play signature:")
print(inspect.signature(music.music_play))

print("\n[TEST] Try play with loops=1")
try:
    music.music_play(str(wav), loops=1)
    time.sleep(2)
    print("✅ SUCCESS with loops=1")
    exit(0)
except Exception as e:
    print("❌ FAIL loops=1:", e)

print("\n[TEST] Try play with loop=1")
try:
    music.music_play(str(wav), loop=1)
    time.sleep(2)
    print("✅ SUCCESS with loop=1")
    exit(0)
except Exception as e:
    print("❌ FAIL loop=1:", e)

print("\n[TEST] Try play without loop argument")
try:
    music.music_play(str(wav))
    time.sleep(2)
    print("✅ SUCCESS without loop")
    exit(0)
except Exception as e:
    print("❌ FAIL without loop:", e)

print("\n❌ All attempts failed")
