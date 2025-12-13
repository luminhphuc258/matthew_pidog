#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pygame

# chạy trong desktop session (AnyDesk / màn hình thật)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ================= FACE DRAW =================

def draw_face(screen, mode: str):
    w, h = screen.get_size()
    screen.fill((245, 230, 215))  # nền sáng giống ảnh

    amber = (255, 170, 0)
    dark = (20, 20, 20)

    # head
    head_r = int(min(w, h) * 0.28)
    cx, cy = w // 2, h // 2 - 30
    pygame.draw.circle(screen, dark, (cx, cy), head_r)

    # eyes base
    eye_w, eye_h = int(head_r * 0.35), int(head_r * 0.45)
    ex1, ex2 = cx - eye_w, cx + eye_w
    ey = cy - eye_h // 4

    def eye(cx, tilt=0, scale=1.0):
        rect = pygame.Rect(
            cx - int(eye_w * scale),
            ey - int(eye_h * scale) + tilt,
            int(eye_w * 2 * scale),
            int(eye_h * 2 * scale),
        )
        pygame.draw.ellipse(screen, amber, rect)
        pygame.draw.ellipse(screen, (255, 210, 120), rect, 3)

    def mouth(y, happy=True):
        width = int(head_r * 0.9)
        height = 14
        mx = cx - width // 2
        pygame.draw.rect(
            screen,
            amber,
            pygame.Rect(mx, y, width, height),
            border_radius=6,
        )

    # ================= MODES =================

    if mode == "friend":
        eye(ex1, 0, 1.0); eye(ex2, 0, 1.0)
        mouth(cy + int(head_r * 0.55))

    elif mode == "follow":
        eye(ex1, 0, 0.9); eye(ex2, 0, 0.9)

    elif mode == "petting":
        eye(ex1, 15, 0.6); eye(ex2, 15, 0.6)

    elif mode == "companion":
        eye(ex1, -5, 0.8); eye(ex2, -5, 0.8)
        mouth(cy + int(head_r * 0.6))

    elif mode == "guardian":
        eye(ex1, 0, 0.7); eye(ex2, 0, 0.7)

    elif mode == "angry":
        eye(ex1, -15, 0.8); eye(ex2, 15, 0.8)
        mouth(cy + int(head_r * 0.5), happy=False)

    pygame.display.flip()


# ================= TEST LOOP =================

def main():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    modes = [
        "friend",
        "follow",
        "petting",
        "companion",
        "guardian",
        "angry",
    ]

    print("Robot Face Test – Ctrl+C to exit")

    while True:
        for m in modes:
            print("FACE:", m)
            t0 = time.time()
            while time.time() - t0 < 2.0:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return
                draw_face(screen, m)
                time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
