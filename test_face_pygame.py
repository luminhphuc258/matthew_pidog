#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import pygame

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


# ===================== Drawing primitives =====================

def aa_circle(screen, color, center, r, width=0):
    pygame.draw.circle(screen, color, center, r, width)

def aa_line(screen, color, p1, p2, width=3):
    pygame.draw.line(screen, color, p1, p2, width)

def aa_ellipse(screen, color, rect, width=0):
    pygame.draw.ellipse(screen, color, rect, width)

def aa_arc(screen, color, rect, start, end, width=4):
    pygame.draw.arc(screen, color, rect, start, end, width)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ===================== Dora-style face renderer =====================

def draw_dora_face(screen, x, y, size, face_id: int):
    """
    face_id: 0..8 (9 faces)
    """
    # palette
    BLUE = (40, 160, 220)
    WHITE = (245, 245, 245)
    BLACK = (25, 25, 25)
    RED = (230, 60, 60)
    YELLOW = (250, 210, 70)
    DARK_RED = (180, 40, 40)

    cx = int(x + size * 0.5)
    cy = int(y + size * 0.48)

    head_r = int(size * 0.38)
    face_r = int(size * 0.30)

    # --- head base (blue) ---
    aa_circle(screen, BLUE, (cx, cy), head_r)

    # --- face area (white) ---
    aa_circle(screen, WHITE, (cx, cy + int(size*0.03)), face_r)

    # --- eyes region parameters ---
    eye_w = int(size * 0.12)
    eye_h = int(size * 0.16)
    eye_gap = int(size * 0.03)

    eye_y = cy - int(size * 0.16)
    left_eye = pygame.Rect(cx - eye_gap - eye_w, eye_y, eye_w, eye_h)
    right_eye = pygame.Rect(cx + eye_gap, eye_y, eye_w, eye_h)

    # --- nose ---
    nose_r = int(size * 0.045)
    nose_c = (cx, cy - int(size * 0.02))
    aa_circle(screen, RED, nose_c, nose_r)
    aa_circle(screen, (255, 255, 255), (nose_c[0]-int(nose_r*0.3), nose_c[1]-int(nose_r*0.3)), max(2, int(nose_r*0.25)))

    # --- center line ---
    aa_line(screen, BLACK, (cx, cy + int(size*0.01)), (cx, cy + int(size*0.20)), 3)

    # --- whiskers (3 each side) ---
    whisk_y0 = cy + int(size*0.02)
    for i in (-1, 0, 1):
        yy = whisk_y0 + i * int(size*0.06)
        aa_line(screen, BLACK, (cx - int(size*0.28), yy), (cx - int(size*0.07), yy), 3)
        aa_line(screen, BLACK, (cx + int(size*0.07), yy), (cx + int(size*0.28), yy), 3)

    # --- collar + bell ---
    collar_y = cy + int(size * 0.36)
    collar_h = int(size * 0.06)
    collar_rect = pygame.Rect(x + int(size*0.18), collar_y, int(size*0.64), collar_h)
    pygame.draw.rect(screen, (220, 40, 40), collar_rect, border_radius=int(collar_h*0.35))

    bell_r = int(size * 0.055)
    bell_c = (cx, collar_y + collar_h + int(size*0.02))
    aa_circle(screen, YELLOW, bell_c, bell_r)
    aa_circle(screen, BLACK, bell_c, bell_r, 2)
    aa_line(screen, BLACK, (bell_c[0]-bell_r, bell_c[1]-int(bell_r*0.15)),
            (bell_c[0]+bell_r, bell_c[1]-int(bell_r*0.15)), 2)
    aa_circle(screen, BLACK, (bell_c[0], bell_c[1]+int(bell_r*0.25)), max(2, int(bell_r*0.12)))

    # ===================== 9 expressions =====================

    def draw_eye_open():
        aa_ellipse(screen, WHITE, left_eye)
        aa_ellipse(screen, WHITE, right_eye)
        aa_ellipse(screen, BLACK, left_eye, 3)
        aa_ellipse(screen, BLACK, right_eye, 3)

        # pupils
        pr = max(3, int(size*0.02))
        aa_circle(screen, BLACK, (left_eye.centerx, left_eye.centery + int(size*0.02)), pr)
        aa_circle(screen, BLACK, (right_eye.centerx, right_eye.centery + int(size*0.02)), pr)

    def draw_eye_happy():
        # closed happy eyes
        lx1 = (left_eye.left + int(eye_w*0.15), left_eye.centery)
        lx2 = (left_eye.right - int(eye_w*0.15), left_eye.centery)
        rx1 = (right_eye.left + int(eye_w*0.15), right_eye.centery)
        rx2 = (right_eye.right - int(eye_w*0.15), right_eye.centery)
        aa_arc(screen, BLACK, pygame.Rect(lx1[0], lx1[1]-int(eye_h*0.2), int(eye_w*0.7), int(eye_h*0.6)), math.pi, 2*math.pi, 4)
        aa_arc(screen, BLACK, pygame.Rect(rx1[0], rx1[1]-int(eye_h*0.2), int(eye_w*0.7), int(eye_h*0.6)), math.pi, 2*math.pi, 4)

    def draw_eye_angry():
        # squint eyes + brows
        draw_eye_open()
        # brows
        aa_line(screen, BLACK,
                (left_eye.left, left_eye.top + int(eye_h*0.2)),
                (left_eye.right, left_eye.top - int(eye_h*0.1)), 4)
        aa_line(screen, BLACK,
                (right_eye.left, right_eye.top - int(eye_h*0.1)),
                (right_eye.right, right_eye.top + int(eye_h*0.2)), 4)

    def draw_mouth_big_smile():
        m_rect = pygame.Rect(cx - int(size*0.16), cy + int(size*0.09), int(size*0.32), int(size*0.22))
        aa_arc(screen, BLACK, m_rect, 0, math.pi, 5)

    def draw_mouth_open():
        m_rect = pygame.Rect(cx - int(size*0.18), cy + int(size*0.10), int(size*0.36), int(size*0.22))
        aa_ellipse(screen, DARK_RED, m_rect)
        aa_ellipse(screen, BLACK, m_rect, 4)
        # tongue
        t_rect = pygame.Rect(m_rect.left + int(m_rect.width*0.25), m_rect.top + int(m_rect.height*0.45),
                             int(m_rect.width*0.5), int(m_rect.height*0.45))
        aa_ellipse(screen, (240, 120, 120), t_rect)

    def draw_mouth_surprise():
        m_rect = pygame.Rect(cx - int(size*0.07), cy + int(size*0.12), int(size*0.14), int(size*0.18))
        aa_ellipse(screen, DARK_RED, m_rect)
        aa_ellipse(screen, BLACK, m_rect, 4)

    def draw_mouth_pout():
        # small o mouth / pout line
        aa_circle(screen, BLACK, (cx, cy + int(size*0.16)), int(size*0.05), 4)

    def draw_sweat():
        # small sweat drop
        drop_c = (cx + int(size*0.26), cy - int(size*0.10))
        aa_ellipse(screen, (180, 220, 255), pygame.Rect(drop_c[0]-8, drop_c[1]-16, 16, 28))
        aa_ellipse(screen, (120, 170, 230), pygame.Rect(drop_c[0]-8, drop_c[1]-16, 16, 28), 2)

    def draw_side_profile():
        # simple side profile overlay (cheat): draw a big blue circle offset + white oval
        # erase and redraw side-ish head
        screen.fill((255, 255, 255), pygame.Rect(x, y, size, size))
        # background light
        pygame.draw.rect(screen, (245, 245, 245), pygame.Rect(x, y, size, size))

        scx = x + int(size*0.48)
        scy = y + int(size*0.52)
        r = int(size*0.38)
        aa_circle(screen, BLUE, (scx, scy), r)
        # white face oval shifted
        face_rect = pygame.Rect(scx - int(r*0.35), scy - int(r*0.30), int(r*0.75), int(r*0.70))
        aa_ellipse(screen, WHITE, face_rect)

        # one eye
        eye_rect = pygame.Rect(scx - int(r*0.05), scy - int(r*0.35), int(r*0.18), int(r*0.22))
        aa_ellipse(screen, WHITE, eye_rect)
        aa_ellipse(screen, BLACK, eye_rect, 3)
        aa_circle(screen, BLACK, (eye_rect.centerx, eye_rect.centery), max(3, int(size*0.02)))

        # nose
        n_c = (scx + int(r*0.25), scy - int(r*0.12))
        aa_circle(screen, RED, n_c, int(size*0.05))
        aa_circle(screen, (255,255,255), (n_c[0]-6, n_c[1]-6), 3)

        # whiskers
        wy = scy - int(r*0.02)
        for i in (-1, 0, 1):
            yy = wy + i * int(size*0.06)
            aa_line(screen, BLACK, (scx + int(r*0.05), yy), (scx + int(r*0.42), yy), 3)

        # mouth small
        aa_arc(screen, BLACK, pygame.Rect(scx + int(r*0.05), scy + int(r*0.02), int(r*0.35), int(r*0.22)),
               0.2*math.pi, 1.0*math.pi, 4)

        # collar
        collar_y2 = scy + int(r*0.38)
        pygame.draw.rect(screen, (220, 40, 40), pygame.Rect(x + int(size*0.22), collar_y2, int(size*0.56), int(size*0.06)),
                         border_radius=10)
        aa_circle(screen, YELLOW, (x + int(size*0.50), collar_y2 + int(size*0.09)), int(size*0.055))
        aa_circle(screen, BLACK, (x + int(size*0.50), collar_y2 + int(size*0.09)), int(size*0.055), 2)

    # Face mapping like the 3x3 image:
    # 0 big laugh, 1 tongue out closed eyes, 2 angry tongue
    # 3 wink laugh, 4 panic scream, 5 side profile
    # 6 pout, 7 cry/sad big mouth, 8 cheeky tongue
    if face_id == 0:
        draw_eye_open()
        draw_mouth_open()
    elif face_id == 1:
        draw_eye_happy()
        # tongue out (mouth open small + tongue)
        m = pygame.Rect(cx - int(size*0.10), cy + int(size*0.10), int(size*0.20), int(size*0.18))
        aa_ellipse(screen, DARK_RED, m)
        aa_ellipse(screen, BLACK, m, 4)
        t = pygame.Rect(m.left + int(m.width*0.35), m.top + int(m.height*0.45), int(m.width*0.30), int(m.height*0.55))
        aa_ellipse(screen, (255, 140, 120), t)
    elif face_id == 2:
        draw_eye_angry()
        draw_mouth_surprise()
    elif face_id == 3:
        # wink + open mouth laugh
        # left eye wink, right eye open
        # left wink
        lx1 = (left_eye.left + int(eye_w*0.15), left_eye.centery)
        aa_arc(screen, BLACK, pygame.Rect(lx1[0], lx1[1]-int(eye_h*0.2), int(eye_w*0.7), int(eye_h*0.6)), math.pi, 2*math.pi, 4)
        # right open
        aa_ellipse(screen, WHITE, right_eye)
        aa_ellipse(screen, BLACK, right_eye, 3)
        aa_circle(screen, BLACK, (right_eye.centerx, right_eye.centery + int(size*0.02)), max(3, int(size*0.02)))
        draw_mouth_open()
    elif face_id == 4:
        # panic scream + sweat
        draw_eye_open()
        # shake lines
        for i in range(4):
            aa_arc(screen, BLACK, pygame.Rect(cx-int(size*0.26), cy-int(size*0.18), int(size*0.18), int(size*0.18)),
                   0.3*math.pi, 1.0*math.pi, 3)
        draw_mouth_open()
        draw_sweat()
    elif face_id == 5:
        draw_side_profile()
        return
    elif face_id == 6:
        # bored/pout
        draw_eye_open()
        draw_mouth_pout()
    elif face_id == 7:
        # crying: sad eyes + big mouth + tear drops
        # sad eyes (half closed)
        aa_arc(screen, BLACK, pygame.Rect(left_eye.left, left_eye.top+int(eye_h*0.25), eye_w, int(eye_h*0.7)), 0, math.pi, 4)
        aa_arc(screen, BLACK, pygame.Rect(right_eye.left, right_eye.top+int(eye_h*0.25), eye_w, int(eye_h*0.7)), 0, math.pi, 4)
        draw_mouth_open()
        # tears
        for dx in (-int(size*0.10), int(size*0.10)):
            drop = pygame.Rect(cx + dx - 8, cy + int(size*0.02), 16, 26)
            aa_ellipse(screen, (180, 220, 255), drop)
            aa_ellipse(screen, (120, 170, 230), drop, 2)
    else:
        # cheeky: one eye bigger + tongue out
        draw_eye_open()
        # tongue out overlay
        t = pygame.Rect(cx - int(size*0.06), cy + int(size*0.18), int(size*0.12), int(size*0.14))
        aa_ellipse(screen, (255, 140, 120), t)
        aa_ellipse(screen, BLACK, t, 3)
        draw_mouth_big_smile()


# ===================== Test: grid 3x3 =====================

def main():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    W, H = screen.get_size()
    bg = (255, 255, 255)

    margin = int(min(W, H) * 0.05)
    grid_w = W - margin * 2
    grid_h = H - margin * 2

    cell = int(min(grid_w / 3, grid_h / 3))
    start_x = (W - cell * 3) // 2
    start_y = (H - cell * 3) // 2

    print("9 faces test (Dora-style). ESC to exit.")
    face_id = 0
    last_switch = time.time()

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                return

        screen.fill(bg)

        # draw 9 faces (0..8)
        for r in range(3):
            for c in range(3):
                idx = r * 3 + c
                x = start_x + c * cell
                y = start_y + r * cell
                draw_dora_face(screen, x, y, cell, idx)

        pygame.display.flip()

        # auto cycle highlight by changing background slightly (optional)
        if time.time() - last_switch > 2.0:
            face_id = (face_id + 1) % 9
            last_switch = time.time()

        time.sleep(0.02)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
