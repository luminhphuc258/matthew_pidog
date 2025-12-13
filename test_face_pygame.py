#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, math
import pygame

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

BLUE = (60, 200, 255)     # neon blue
BLUE2 = (120, 230, 255)   # highlight
BLACK = (0, 0, 0)

EMOS = [
    "happy_open",    # 1
    "happy_closed",  # 2
    "love",          # 3
    "confused",      # 4
    "surprised",     # 5
    "talking",       # 6
    "laugh",         # 7
    "sleep",         # 8
    "sad",           # 9
    "pain",          # 10
    "dead",          # 11
    "angry",         # 12
]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_round_rect(surf, color, rect, radius, width=0):
    pygame.draw.rect(surf, color, rect, width, border_radius=radius)

def draw_text(screen, text, x, y, size, color=BLUE):
    font = pygame.font.SysFont("DejaVu Sans", size, bold=True)
    img = font.render(text, True, color)
    r = img.get_rect(center=(x, y))
    screen.blit(img, r)

def eye_open(screen, cx, cy, w, h, pupil_dx=0, pupil_dy=0, blink=0.0):
    """
    blink: 0..1 (0 open, 1 closed)
    """
    # eyelid effect by shrinking height
    hh = max(6, int(h * (1.0 - 0.92 * blink)))
    rect = pygame.Rect(cx - w//2, cy - hh//2, w, hh)

    # eye outline + fill
    draw_round_rect(screen, BLUE, rect, radius=hh//2, width=4)
    # pupil
    pr = max(5, int(min(w, hh) * 0.13))
    px = clamp(cx + pupil_dx, rect.left + pr + 6, rect.right - pr - 6)
    py = clamp(cy + pupil_dy, rect.top + pr + 6, rect.bottom - pr - 6)
    pygame.draw.circle(screen, BLUE, (px, py), pr)
    pygame.draw.circle(screen, BLUE2, (px - pr//3, py - pr//3), max(2, pr//3))

def eye_happy_arc(screen, cx, cy, w, h):
    # smiling closed eye arc
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    pygame.draw.arc(screen, BLUE, rect, math.pi, 2*math.pi, 6)

def eye_x(screen, cx, cy, size):
    s = size
    pygame.draw.line(screen, BLUE, (cx - s, cy - s), (cx + s, cy + s), 6)
    pygame.draw.line(screen, BLUE, (cx - s, cy + s), (cx + s, cy - s), 6)

def eye_spiral(screen, cx, cy, r, t):
    # spiral using polyline
    pts = []
    turns = 3.5
    for i in range(80):
        a = (i/80.0) * turns * 2*math.pi + t*2.2
        rr = (i/80.0) * r
        pts.append((cx + rr*math.cos(a), cy + rr*math.sin(a)))
    pygame.draw.lines(screen, BLUE, False, pts, 4)

def mouth_smile(screen, cx, cy, w, h, open_amount=0.0):
    # smile arc + optional open
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    pygame.draw.arc(screen, BLUE, rect, 0, math.pi, 6)
    if open_amount > 0:
        ow = int(w*0.25)
        oh = int(h*0.35*open_amount)
        if oh > 3:
            pygame.draw.ellipse(screen, BLUE, pygame.Rect(cx-ow//2, cy+int(h*0.15), ow, oh), 4)

def mouth_open(screen, cx, cy, w, h):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    draw_round_rect(screen, BLUE, rect, radius=h//2, width=6)

def mouth_sad(screen, cx, cy, w, h):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    pygame.draw.arc(screen, BLUE, rect, math.pi, 2*math.pi, 6)

def mouth_flat(screen, cx, cy, w):
    pygame.draw.line(screen, BLUE, (cx - w//2, cy), (cx + w//2, cy), 6)

def mouth_wavy(screen, cx, cy, w, amp, t):
    pts=[]
    for i in range(50):
        x = cx - w//2 + (i/49.0)*w
        y = cy + math.sin(t*7 + i*0.35)*amp
        pts.append((x,y))
    pygame.draw.lines(screen, BLUE, False, pts, 6)

def bubble(screen, x, y, w, h, text, blink_on=True):
    # simple speech bubble
    rect = pygame.Rect(x - w//2, y - h//2, w, h)
    draw_round_rect(screen, BLUE, rect, radius=18, width=4)
    if blink_on:
        draw_text(screen, text, x, y, size=int(h*0.55), color=BLUE)

def hearts(screen, cx, cy, t):
    # two hearts pulsing
    pulse = 1.0 + 0.12*math.sin(t*4.0)
    for dx in (-70, 70):
        r = int(26*pulse)
        x = cx + dx
        y = cy - 20
        # heart made of 2 circles + triangle-ish polygon
        pygame.draw.circle(screen, BLUE, (x - r//2, y), r//2)
        pygame.draw.circle(screen, BLUE, (x + r//2, y), r//2)
        pts = [(x - r, y), (x + r, y), (x, y + int(r*1.4))]
        pygame.draw.polygon(screen, BLUE, pts)

def ex_mark(screen, cx, cy, t):
    # ! bouncing
    bounce = int(8*abs(math.sin(t*3)))
    draw_text(screen, "!", cx, cy - 120 - bounce, 64, BLUE)

def q_mark(screen, cx, cy, t):
    # ? wobble
    wob = int(6*math.sin(t*3))
    draw_text(screen, "?", cx, cy - 120 + wob, 64, BLUE)

def zzz(screen, cx, cy, t):
    # Zzz floating
    for i in range(3):
        yy = cy - 140 - i*40 - int(10*math.sin(t*2 + i))
        xx = cx + 120 + i*28
        draw_text(screen, "Z", xx, yy, 42 - i*6, BLUE)

def tear(screen, x, y, t):
    # tear drop bobbing
    bob = int(6*math.sin(t*3))
    rect = pygame.Rect(x-10, y+15+bob, 20, 34)
    pygame.draw.ellipse(screen, BLUE, rect, 3)

# ===================== 12 face animations =====================

def render_face(screen, emo, t):
    W, H = screen.get_size()
    screen.fill(BLACK)

    cx, cy = W//2, H//2

    # scale big
    eye_y = cy - int(H*0.10)
    mouth_y = cy + int(H*0.16)

    eye_w = int(W*0.22)
    eye_h = int(H*0.16)
    gap = int(W*0.12)

    lx = cx - gap
    rx = cx + gap

    mouth_w = int(W*0.38)
    mouth_h = int(H*0.18)

    # animation common
    blink = 0.0
    # periodic blink (short)
    phase = (t % 3.5)
    if phase < 0.12:
        blink = 1.0 - phase/0.12
    elif phase < 0.24:
        blink = (phase-0.12)/0.12
    else:
        blink = 0.0
    blink = clamp(blink, 0.0, 1.0)

    # pupils drifting
    pdx = int(math.sin(t*1.8) * (W*0.01))
    pdy = int(math.sin(t*1.2 + 1.2) * (H*0.01))

    if emo == "happy_open":
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx, pdy, blink)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, pdy, blink)
        mouth_smile(screen, cx, mouth_y, mouth_w, mouth_h, open_amount=0.25+0.15*math.sin(t*4))

    elif emo == "happy_closed":
        eye_happy_arc(screen, lx, eye_y, int(eye_w*0.9), int(eye_h*0.9))
        eye_happy_arc(screen, rx, eye_y, int(eye_w*0.9), int(eye_h*0.9))
        mouth_smile(screen, cx, mouth_y, mouth_w, mouth_h, open_amount=0.15)

    elif emo == "love":
        # heart eyes
        hearts(screen, cx, eye_y, t)
        mouth_smile(screen, cx, mouth_y, mouth_w, mouth_h, open_amount=0.22)

    elif emo == "confused":
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy, blink*0.4)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, -pdy, blink*0.4)
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.55))
        q_mark(screen, cx, cy, t)

    elif emo == "surprised":
        eye_open(screen, lx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        eye_open(screen, rx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        mouth_open(screen, cx, mouth_y, int(mouth_w*0.28), int(mouth_h*0.22))
        ex_mark(screen, cx, cy, t)

    elif emo == "talking":
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx, pdy, blink)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, pdy, blink)
        # mouth talk (open/close)
        amt = 0.35 + 0.35*(0.5+0.5*math.sin(t*8.0))
        mouth_open(screen, cx, mouth_y, int(mouth_w*0.22), int(mouth_h*0.22*amt))
        bubble(screen, cx+int(W*0.22), cy-int(H*0.23), int(W*0.18), int(H*0.12), "...", blink_on=(math.sin(t*3)>-0.2))

    elif emo == "laugh":
        # squint laughing
        eye_happy_arc(screen, lx, eye_y, int(eye_w*0.9), int(eye_h*0.9))
        eye_happy_arc(screen, rx, eye_y, int(eye_w*0.9), int(eye_h*0.9))
        mouth_smile(screen, cx, mouth_y, mouth_w, mouth_h, open_amount=0.55+0.15*math.sin(t*5.0))

    elif emo == "sleep":
        # sleepy eyes + zzz
        eye_happy_arc(screen, lx, eye_y+20, int(eye_w*0.85), int(eye_h*0.85))
        eye_happy_arc(screen, rx, eye_y+20, int(eye_w*0.85), int(eye_h*0.85))
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.35))
        zzz(screen, cx, cy, t)

    elif emo == "sad":
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy+10, blink*0.2)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, pdy+10, blink*0.2)
        mouth_sad(screen, cx, mouth_y+10, mouth_w, mouth_h)
        tear(screen, lx+int(eye_w*0.18), eye_y+int(eye_h*0.05), t)

    elif emo == "pain":
        # tight eyes + wavy mouth (shake)
        # eyes squeezed
        eye_happy_arc(screen, lx, eye_y+10, int(eye_w*0.85), int(eye_h*0.60))
        eye_happy_arc(screen, rx, eye_y+10, int(eye_w*0.85), int(eye_h*0.60))
        mouth_wavy(screen, cx, mouth_y, int(mouth_w*0.55), amp=int(H*0.018), t=t)
        # little "stress lines"
        for i in range(6):
            x = cx - int(W*0.20) + i*int(W*0.07)
            pygame.draw.line(screen, BLUE, (x, eye_y-int(H*0.12)), (x+10, eye_y-int(H*0.15)), 3)

    elif emo == "dead":
        eye_x(screen, lx, eye_y, int(W*0.03))
        eye_x(screen, rx, eye_y, int(W*0.03))
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.40))

    elif emo == "angry":
        # sharp angry eyes (squint + pupils jitter)
        jitter = int(math.sin(t*12)*6)
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx+jitter, pdy, blink*0.1)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx-jitter, pdy, blink*0.1)
        # eyebrows
        pygame.draw.line(screen, BLUE, (lx-int(eye_w*0.55), eye_y-int(eye_h*0.55)),
                         (lx+int(eye_w*0.55), eye_y-int(eye_h*0.15)), 7)
        pygame.draw.line(screen, BLUE, (rx-int(eye_w*0.55), eye_y-int(eye_h*0.15)),
                         (rx+int(eye_w*0.55), eye_y-int(eye_h*0.55)), 7)
        mouth_wavy(screen, cx, mouth_y+10, int(mouth_w*0.55), amp=int(H*0.010), t=t)

    pygame.display.flip()


# ===================== Main test app =====================

def main():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    idx = 0
    auto = True
    last = time.time()

    print("12 animated faces | 1..12 select | SPACE auto | ESC quit")

    while True:
        now = time.time()
        t = now

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if e.key == pygame.K_SPACE:
                    auto = not auto

                # 1..9
                if pygame.K_1 <= e.key <= pygame.K_9:
                    idx = e.key - pygame.K_1
                    auto = False
                # 0 = 10
                if e.key == pygame.K_0:
                    idx = 9
                    auto = False
                # - = 11, = = 12  (fallback mapping on small keyboard)
                if e.key == pygame.K_MINUS:
                    idx = 10
                    auto = False
                if e.key == pygame.K_EQUALS:
                    idx = 11
                    auto = False

        if auto and (now - last) > 2.0:
            idx = (idx + 1) % len(EMOS)
            last = now

        render_face(screen, EMOS[idx], t)
        time.sleep(0.016)  # ~60fps


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
