#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from face_display import FaceDisplay   # nếu class nằm trong face_display.py

def main():
    print("[TEST] Init FaceDisplay (framebuffer)...")
    face = FaceDisplay()   # auto detect /dev/fb0, size, bpp

    emotions = ["neutral", "happy", "angry", "sad"]

    print("[TEST] Cycling emotions...")
    while True:
        for emo in emotions:
            print(f"[FACE] {emo}")
            face.set_emotion(emo)

            # vẽ liên tục trong 2 giây để chắc chắn framebuffer được refresh
            t0 = time.time()
            while time.time() - t0 < 2.0:
                face.tick()
                time.sleep(0.05)  # ~20 FPS

if __name__ == "__main__":
    main()
