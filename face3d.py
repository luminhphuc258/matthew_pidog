#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import pygame

# ---------------------------
# Helpers: fake 3D shading
# ---------------------------

def clamp(x, a, b):
    return a if x < a else b if x > b else x

def draw_radial_gradient_circle(surf, center, radius, inner_color, outer_color, steps=64):
    """Fake 3D: vẽ circle nhiều vòng để tạo radial gradient."""
    cx, cy = center
    for i in range(steps, 0, -1):
        t = i / steps
        r = int(radius * t)
        col = (
            int(outer_color[0] + (inner_color[0] - outer_color[0]) * t),
            int(outer_color[1] + (inner_color[1] - outer_color[1]) * t),
            int(outer_color[2] + (inner_color[2] - outer_color[2]) * t),
            int(outer_color[3] + (inner_color[3] - outer_color[3]) * t),
        )
        pygame.draw.circle(surf, col, (cx, cy), r)

def draw_glow_circle(surf, center, radius, color, glow=18):
    """Glow: vẽ nhiều vòng alpha nhỏ dần."""
    cx, cy = center
    for i in range(glow, 0, -1):
        a = int(color[3] * (i / glow) * 0.35)
        pygame.draw.circle(surf, (color[0], color[1], color[2], a), (cx, cy), radius + i)

def draw_glow_line(surf, p1, p2, color, width=6, glow=14):
    """Glow cho line (miệng)."""
    for i in range(glow, 0, -1):
        a = int(color[3] * (i / glow) * 0.25)
        pygame.draw.line(surf, (color[0], color[1], color[2], a), p1, p2, width + i*2)
    pygame.draw.line(surf, color, p1, p2, width)

def draw_eye_3d(base, pos, eye_r, iris_r, pupil_r, neon):
    """
    Vẽ 1 mắt 3D giả lập:
    - eyeball: radial gradient (nổi khối)
    - highlight: chấm sáng
    - iris+pupil: pupil di chuyển
    """
    x, y = pos

    # glow ngoài
    draw_glow_circle(base, (x, y), eye_r, neon, glow=22)

    # eyeball 3D (tối ngoài, sáng trong)
    eyeball = pygame.Surface((eye_r*2+4, eye_r*2+4), pygame.SRCALPHA)
    draw_radial_gradient_circle(
        eyeball,
        (eye_r+2, eye_r+2),
        eye_r,
        inner_color=(30, 120, 255, 210),
        outer_color=(0, 10, 35, 220),
        steps=80
    )
    base.blit(eyeball, (x-eye_r-2, y-eye_r-2))

    # viền neon (thick)
    pygame.draw.circle(base, (neon[0], neon[1], neon[2], 220), (x, y), eye_r, 5)

    # highlight (ánh sáng)
    pygame.draw.circle(base, (255, 255, 255, 90), (x-int(eye_r*0.35), y-int(eye_r*0.35)), int(eye_r*0.18))
    pygame.draw.circle(base, (255, 255, 255, 40), (x-int(eye_r*0.20), y-int(eye_r*0.20)), int(eye_r*0.10))

def draw_iris_and_pupil(base, eye_center, eye_r, iris_r, pupil_r, look_target, neon):
    ex, ey = eye_center
    tx, ty = look_target

    # vector nhìn
    vx, vy = tx - ex, ty - ey
    dist = math.hypot(vx, vy) or 1.0
    nx, ny = vx / dist, vy / dist

    # giới hạn pupil trong mắt
    max_offset = eye_r - iris_r - 8
    px = ex + int(nx * max_offset)
    py = ey + int(ny * max_offset)

    # iris (gradient nhỏ)
    iris = pygame.Surface((iris_r*2+2, iris_r*2+2), pygame.SRCALPHA)
    draw_radial_gradient_circle(
        iris,
        (iris_r+1, iris_r+1),
        iris_r,
        inner_color=(neon[0], neon[1], neon[2], 200),
        outer_color=(0, 0, 0, 220),
        steps=60
    )
    base.blit(iris, (px-iris_r-1, py-iris_r-1))

    # pupil
    pygame.draw.circle(base, (0, 0, 0, 230), (px, py), pupil_r)

    # pupil highlight
    pygame.draw.circle(base, (255, 255, 255, 80), (px-int(pupil_r*0.35), py-int(pupil_r*0.35)), int(pupil_r*0.30))

def draw_mouth_3d(base, rect, mood, neon):
    """
    Vẽ miệng kiểu neon 3D giả lập.
    mood: "smile" | "sad" | "neutral"
    """
    x, y, w, h = rect
    cx = x + w//2
    cy = y + h//2

    # glow nền miệng
    mouth_layer = pygame.Surface((w, h), pygame.SRCALPHA)

    if mood == "smile":
        # cung cười
        pts = []
        for i in range(0, 31):
            t = i / 30
            px = x + int(t * w)
            # parabol cười
            py = y + int(h*0.55 + (t-0.5)*(t-0.5) * h*1.0)
            pts.append((px, py))
        for i in range(len(pts)-1):
            draw_glow_line(base, pts[i], pts[i+1], neon, width=8, glow=14)

    elif mood == "sad":
        # cung buồn (úp ngược)
        pts = []
        for i in range(0, 31):
            t = i / 30
            px = x + int(t * w)
            py = y + int(h*0.45 - (t-0.5)*(t-0.5) * h*1.0)
            pts.append((px, py))
        for i in range(len(pts)-1):
            draw_glow_line(base, pts[i], pts[i+1], neon, width=8, glow=14)

    else:
        # ngang
        p1 = (x + int(w*0.15), cy)
        p2 = (x + int(w*0.85), cy)
        draw_glow_line(base, p1, p2, neon, width=10, glow=14)

# ---------------------------
# Main demo
# ---------------------------

def main():
    pygame.init()

    # Fullscreen trên Pi
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H = screen.get_size()
    clock = pygame.time.Clock()

    # Màu nền + neon
    bg = (0, 0, 0)
    neon = (40, 140, 255, 255)  # xanh dương robot

    # Layout mắt to full screen
    eye_r = int(min(W, H) * 0.17)
    iris_r = int(eye_r * 0.42)
    pupil_r = int(eye_r * 0.18)

    left_eye = (W//2 - int(eye_r*1.6), H//2 - int(eye_r*0.2))
    right_eye = (W//2 + int(eye_r*1.6), H//2 - int(eye_r*0.2))

    mouth_rect = (W//2 - int(W*0.20), H//2 + int(H*0.20), int(W*0.40), int(H*0.18))

    mood = "smile"
    t = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        t += dt

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                if e.key == pygame.K_1:
                    mood = "smile"
                if e.key == pygame.K_2:
                    mood = "neutral"
                if e.key == pygame.K_3:
                    mood = "sad"

        screen.fill(bg)

        # target nhìn: bay vòng vòng để thấy mắt chuyển động
        target_x = W//2 + int(math.cos(t*1.4) * W*0.18)
        target_y = H//2 + int(math.sin(t*1.1) * H*0.14)

        # Vẽ eyeball 3D trước
        draw_eye_3d(screen, left_eye, eye_r, iris_r, pupil_r, neon)
        draw_eye_3d(screen, right_eye, eye_r, iris_r, pupil_r, neon)

        # Vẽ iris+pupil sau
        draw_iris_and_pupil(screen, left_eye, eye_r, iris_r, pupil_r, (target_x, target_y), neon)
        draw_iris_and_pupil(screen, right_eye, eye_r, iris_r, pupil_r, (target_x, target_y), neon)

        # Miệng 3D neon
        draw_mouth_3d(screen, mouth_rect, mood, neon)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
