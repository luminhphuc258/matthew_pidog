#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

# Ưu tiên chạy không cần X11: KMSDRM -> FBCON -> DIRECTFB
# (phải set trước khi import pygame)
if "DISPLAY" not in os.environ:
    os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame


def draw_face(screen, emo: str):
    w, h = screen.get_size()
    screen.fill((20, 20, 20))  # background

    amber = (255, 170, 0)

    # eye positions
    cx1, cx2 = int(w * 0.35), int(w * 0.65)
    cy = int(h * 0.45)
    eye_w, eye_h = int(w * 0.08), int(h * 0.14)

    def eye(cx, tilt=0):
        y_shift = int(tilt / 2)
        rect = pygame.Rect(cx - eye_w, cy - eye_h + y_shift, eye_w * 2, eye_h * 2)
        pygame.draw.rect(screen, amber, rect, border_radius=8)

    # mouth
    def mouth_line(y, thickness=10):
        mx0, mx1 = int(w * 0.40), int(w * 0.60)
        pygame.draw.rect(screen, amber, pygame.Rect(mx0, y, mx1 - mx0, thickness), border_radius=6)

    if emo == "happy":
        eye(cx1, 0); eye(cx2, 0)
        mouth_line(int(h * 0.72), thickness=10)
    elif emo == "sad":
        eye(cx1, -10); eye(cx2, 10)
        mouth_line(int(h * 0.75), thickness=10)
    elif emo == "angry":
        eye(cx1, 18); eye(cx2, -18)
        mouth_line(int(h * 0.68), thickness=14)
    else:
        eye(cx1, 0); eye(cx2, 0)

    pygame.display.flip()


def main():
    print("[TEST] init pygame...")
    pygame.init()

    # fullscreen theo màn hình thật (WaveShare)
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    emotions = ["neutral", "happy", "angry", "sad"]

    print("[TEST] cycling face... (Ctrl+C to stop)")
    while True:
        for emo in emotions:
            print("FACE:", emo)
            t0 = time.time()
            while time.time() - t0 < 2.0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                draw_face(screen, emo)
                time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
