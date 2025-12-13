#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, math
import pygame

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ===== STYLE =====
BG = (0, 0, 0)
BLUE = (70, 210, 255)     # neon core
BLUE2 = (140, 240, 255)   # highlight glow

# Thickness (đậm như hình bạn gửi)
THICK_LINE = 10
THICK_EYE = 10
THICK_MOUTH = 12

# Faces list: 12 old + 3 new
EMOS = [
    "happy_open",    # 1
    "happy_closed",  # 2
    "love_old",      # 3
    "confused",      # 4
    "surprised",     # 5
    "talking",       # 6
    "laugh",         # 7
    "sleep",         # 8
    "sad",           # 9
    "pain",          # 10
    "dead",          # 11
    "angry",         # 12
    "very_happy",    # 13
    "very_sad",      # 14
    "love",          # 15
]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_round_rect(surf, color, rect, radius, width=0):
    pygame.draw.rect(surf, color, rect, width, border_radius=radius)

def draw_glow_line(screen, p1, p2, width):
    # simple glow by drawing thicker faint lines behind
    pygame.draw.line(screen, BLUE2, p1, p2, max(1, width+8))
    pygame.draw.line(screen, BLUE,  p1, p2, width)

def draw_glow_arc(screen, rect, start, end, width):
    pygame.draw.arc(screen, BLUE2, rect, start, end, max(1, width+8))
    pygame.draw.arc(screen, BLUE,  rect, start, end, width)

def draw_glow_rect_outline(screen, rect, radius, width):
    # outline with glow
    pygame.draw.rect(screen, BLUE2, rect, max(1, width+8), border_radius=radius)
    pygame.draw.rect(screen, BLUE,  rect, width, border_radius=radius)

def draw_glow_circle(screen, center, r, width):
    pygame.draw.circle(screen, BLUE2, center, r, max(1, width+8))
    pygame.draw.circle(screen, BLUE,  center, r, width)

def eye_open(screen, cx, cy, w, h, pupil_dx=0, pupil_dy=0, blink=0.0):
    # eyelid effect by shrinking height
    hh = max(10, int(h * (1.0 - 0.92 * blink)))
    rect = pygame.Rect(cx - w//2, cy - hh//2, w, hh)
    radius = max(10, hh//2)

    # thick outline
    draw_glow_rect_outline(screen, rect, radius=radius, width=THICK_EYE)

    # pupil
    pr = max(10, int(min(w, hh) * 0.16))
    px = clamp(cx + pupil_dx, rect.left + pr + 10, rect.right - pr - 10)
    py = clamp(cy + pupil_dy, rect.top + pr + 10, rect.bottom - pr - 10)

    pygame.draw.circle(screen, BLUE2, (px, py), pr+6)
    pygame.draw.circle(screen, BLUE,  (px, py), pr)
    pygame.draw.circle(screen, (255,255,255), (px - pr//3, py - pr//3), max(3, pr//3))

def eye_happy_arc(screen, cx, cy, w, h):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    draw_glow_arc(screen, rect, math.pi, 2*math.pi, THICK_LINE)

def eye_sad_arc(screen, cx, cy, w, h):
    # downturned arc (very sad style)
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    draw_glow_arc(screen, rect, 0, math.pi, THICK_LINE)

def eye_x(screen, cx, cy, s):
    draw_glow_line(screen, (cx - s, cy - s), (cx + s, cy + s), THICK_LINE)
    draw_glow_line(screen, (cx - s, cy + s), (cx + s, cy - s), THICK_LINE)

def hearts_eyes(screen, cx, cy, t, big=True):
    pulse = 1.0 + 0.10*math.sin(t*4.0)
    base = 42 if big else 30
    r = int(base * pulse)

    for dx in (-90, 90):
        x = cx + dx
        y = cy
        # heart = 2 circles + polygon
        pygame.draw.circle(screen, BLUE2, (x - r//2, y), r//2 + 6)
        pygame.draw.circle(screen, BLUE2, (x + r//2, y), r//2 + 6)
        pygame.draw.polygon(screen, BLUE2, [(x - r, y), (x + r, y), (x, y + int(r*1.45))])

        pygame.draw.circle(screen, BLUE, (x - r//2, y), r//2)
        pygame.draw.circle(screen, BLUE, (x + r//2, y), r//2)
        pygame.draw.polygon(screen, BLUE, [(x - r, y), (x + r, y), (x, y + int(r*1.45))])

def mouth_smile(screen, cx, cy, w, h, open_amt=0.0):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    draw_glow_arc(screen, rect, 0, math.pi, THICK_MOUTH)
    if open_amt > 0:
        ow = int(w*0.28)
        oh = int(h*0.50*open_amt)
        if oh > 10:
            r = max(10, oh//2)
            rr = pygame.Rect(cx-ow//2, cy+int(h*0.12), ow, oh)
            draw_glow_rect_outline(screen, rr, radius=r, width=THICK_LINE)

def mouth_open(screen, cx, cy, w, h):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    r = max(12, h//2)
    draw_glow_rect_outline(screen, rect, radius=r, width=THICK_MOUTH)

def mouth_sad(screen, cx, cy, w, h):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    draw_glow_arc(screen, rect, math.pi, 2*math.pi, THICK_MOUTH)

def mouth_flat(screen, cx, cy, w):
    draw_glow_line(screen, (cx - w//2, cy), (cx + w//2, cy), THICK_MOUTH)

def bubble(screen, x, y, w, h, t):
    rect = pygame.Rect(x - w//2, y - h//2, w, h)
    draw_glow_rect_outline(screen, rect, radius=22, width=THICK_LINE)
    # blinking dots
    on = (math.sin(t*3.2) > -0.2)
    if on:
        for i in (-1, 0, 1):
            draw_glow_circle(screen, (x + i*26, y), 7, 0)

def ex_mark(screen, cx, cy, t):
    bounce = int(10*abs(math.sin(t*3)))
    # draw "!" using lines thick
    top = (cx, cy - 150 - bounce)
    mid = (cx, cy - 95 - bounce)
    draw_glow_line(screen, top, mid, THICK_LINE)
    draw_glow_circle(screen, (cx, cy - 70 - bounce), 7, 0)

def q_mark(screen, cx, cy, t):
    wob = int(8*math.sin(t*3))
    # "?" simplified: arc + dot
    rect = pygame.Rect(cx-30, cy-175+wob, 60, 70)
    draw_glow_arc(screen, rect, math.pi*1.1, math.pi*2.2, THICK_LINE)
    draw_glow_line(screen, (cx+28, cy-110+wob), (cx+28, cy-90+wob), THICK_LINE)
    draw_glow_circle(screen, (cx+28, cy-70+wob), 7, 0)

def zzz(screen, cx, cy, t):
    # draw Z Z Z as thick lines (no font)
    for i in range(3):
        yy = cy - 170 - i*45 - int(10*math.sin(t*2 + i))
        xx = cx + 170 + i*30
        # Z
        draw_glow_line(screen, (xx-18, yy-14), (xx+18, yy-14), THICK_LINE)
        draw_glow_line(screen, (xx+18, yy-14), (xx-18, yy+14), THICK_LINE)
        draw_glow_line(screen, (xx-18, yy+14), (xx+18, yy+14), THICK_LINE)

def tears(screen, x, y, t):
    bob = int(8*math.sin(t*3))
    rect = pygame.Rect(x-10, y+20+bob, 20, 40)
    pygame.draw.ellipse(screen, BLUE2, rect, 8)
    pygame.draw.ellipse(screen, BLUE,  rect, 4)

def mouth_wavy(screen, cx, cy, w, amp, t):
    pts=[]
    for i in range(48):
        x = cx - w//2 + (i/47.0)*w
        y = cy + math.sin(t*7 + i*0.35)*amp
        pts.append((x,y))
    # glow polyline: draw twice
    pygame.draw.lines(screen, BLUE2, False, pts, THICK_MOUTH+6)
    pygame.draw.lines(screen, BLUE,  False, pts, THICK_MOUTH)

# ===================== render =====================

def render_face(screen, emo, t):
    W, H = screen.get_size()
    screen.fill(BG)

    cx, cy = W//2, H//2

    # Make face big & centered (full screen feeling)
    eye_y = cy - int(H*0.12)
    mouth_y = cy + int(H*0.18)

    eye_w = int(W*0.26)
    eye_h = int(H*0.18)
    gap = int(W*0.14)

    lx = cx - gap
    rx = cx + gap

    mouth_w = int(W*0.44)
    mouth_h = int(H*0.22)

    # blink
    phase = (t % 3.2)
    blink = 0.0
    if phase < 0.10:
        blink = 1.0 - phase/0.10
    elif phase < 0.20:
        blink = (phase-0.10)/0.10
    blink = clamp(blink, 0.0, 1.0)

    pdx = int(math.sin(t*1.8) * (W*0.012))
    pdy = int(math.sin(t*1.2 + 1.2) * (H*0.010))

    if emo == "happy_open":
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx, pdy, blink)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, pdy, blink)
        mouth_smile(screen, cx, mouth_y, mouth_w, mouth_h, open_amt=0.25+0.15*math.sin(t*4))

    elif emo == "happy_closed":
        eye_happy_arc(screen, lx, eye_y, int(eye_w*0.95), int(eye_h*0.95))
        eye_happy_arc(screen, rx, eye_y, int(eye_w*0.95), int(eye_h*0.95))
        mouth_smile(screen, cx, mouth_y, mouth_w, mouth_h, open_amt=0.18)

    elif emo == "love_old":
        hearts_eyes(screen, cx, eye_y, t, big=False)
        mouth_smile(screen, cx, mouth_y, mouth_w, mouth_h, open_amt=0.22)

    elif emo == "confused":
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy, blink*0.4)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, -pdy, blink*0.4)
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.55))
        q_mark(screen, cx, cy, t)

    elif emo == "surprised":
        eye_open(screen, lx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        eye_open(screen, rx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        mouth_open(screen, cx, mouth_y, int(mouth_w*0.22), int(mouth_h*0.22))
        ex_mark(screen, cx, cy, t)

    elif emo == "talking":
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx, pdy, blink)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, pdy, blink)
        amt = 0.35 + 0.35*(0.5+0.5*math.sin(t*8.0))
        mouth_open(screen, cx, mouth_y, int(mouth_w*0.22), int(mouth_h*0.22*amt))
        bubble(screen, cx+int(W*0.22), cy-int(H*0.25), int(W*0.20), int(H*0.14), t)

    elif emo == "laugh":
        eye_happy_arc(screen, lx, eye_y, int(eye_w*0.95), int(eye_h*0.95))
        eye_happy_arc(screen, rx, eye_y, int(eye_w*0.95), int(eye_h*0.95))
        mouth_smile(screen, cx, mouth_y, mouth_w, mouth_h, open_amt=0.55+0.15*math.sin(t*5.0))

    elif emo == "sleep":
        eye_happy_arc(screen, lx, eye_y+18, int(eye_w*0.90), int(eye_h*0.80))
        eye_happy_arc(screen, rx, eye_y+18, int(eye_w*0.90), int(eye_h*0.80))
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.35))
        zzz(screen, cx, cy, t)

    elif emo == "sad":
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy+12, blink*0.2)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, pdy+12, blink*0.2)
        mouth_sad(screen, cx, mouth_y+10, mouth_w, mouth_h)
        tears(screen, lx+int(eye_w*0.20), eye_y+int(eye_h*0.05), t)

    elif emo == "pain":
        eye_happy_arc(screen, lx, eye_y+10, int(eye_w*0.90), int(eye_h*0.55))
        eye_happy_arc(screen, rx, eye_y+10, int(eye_w*0.90), int(eye_h*0.55))
        mouth_wavy(screen, cx, mouth_y, int(mouth_w*0.60), amp=int(H*0.020), t=t)

    elif emo == "dead":
        eye_x(screen, lx, eye_y, int(W*0.035))
        eye_x(screen, rx, eye_y, int(W*0.035))
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.40))

    elif emo == "angry":
        jitter = int(math.sin(t*12)*8)
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx+jitter, pdy, blink*0.1)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx-jitter, pdy, blink*0.1)
        # brows thick
        draw_glow_line(screen,
                       (lx-int(eye_w*0.55), eye_y-int(eye_h*0.55)),
                       (lx+int(eye_w*0.55), eye_y-int(eye_h*0.15)), THICK_LINE+2)
        draw_glow_line(screen,
                       (rx-int(eye_w*0.55), eye_y-int(eye_h*0.15)),
                       (rx+int(eye_w*0.55), eye_y-int(eye_h*0.55)), THICK_LINE+2)
        mouth_wavy(screen, cx, mouth_y+10, int(mouth_w*0.60), amp=int(H*0.012), t=t)

    # ===== NEW 3 FACES (style like your samples) =====
    elif emo == "very_happy":
        # eyes are big curved "U" + smile
        eye_happy_arc(screen, lx, eye_y, int(eye_w*1.05), int(eye_h*1.05))
        eye_happy_arc(screen, rx, eye_y, int(eye_w*1.05), int(eye_h*1.05))
        mouth_smile(screen, cx, mouth_y, int(mouth_w*0.70), int(mouth_h*0.85),
                    open_amt=0.15 + 0.10*math.sin(t*4.5))

    elif emo == "very_sad":
        # droopy eyes (down arcs) + sad mouth
        eye_sad_arc(screen, lx, eye_y+10, int(eye_w*1.05), int(eye_h*0.95))
        eye_sad_arc(screen, rx, eye_y+10, int(eye_w*1.05), int(eye_h*0.95))
        mouth_sad(screen, cx, mouth_y+20, int(mouth_w*0.72), int(mouth_h*0.90))
        # subtle tears both sides
        tears(screen, lx+int(eye_w*0.20), eye_y+int(eye_h*0.05), t)
        tears(screen, rx+int(eye_w*0.20), eye_y+int(eye_h*0.05), t+0.6)

    elif emo == "love":
        # love eyes as two filled circles (like your love sample could be round)
        pulse = 1.0 + 0.12*math.sin(t*4.0)
        r = int(min(W, H)*0.055*pulse)
        pygame.draw.circle(screen, BLUE2, (lx, eye_y), r+8)
        pygame.draw.circle(screen, BLUE2, (rx, eye_y), r+8)
        pygame.draw.circle(screen, BLUE,  (lx, eye_y), r)
        pygame.draw.circle(screen, BLUE,  (rx, eye_y), r)
        # small highlight
        pygame.draw.circle(screen, (255,255,255), (lx-int(r*0.3), eye_y-int(r*0.3)), max(4, r//3))
        pygame.draw.circle(screen, (255,255,255), (rx-int(r*0.3), eye_y-int(r*0.3)), max(4, r//3))
        mouth_smile(screen, cx, mouth_y, int(mouth_w*0.62), int(mouth_h*0.80), open_amt=0.12)
        # small floating hearts
        hearts_eyes(screen, cx, eye_y-int(H*0.18), t, big=False)

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    idx = 0
    auto = True
    last = time.time()

    print("15 thick faces | 1..9 | 0=10 | -=11 | ==12 | [=13 | ]=14 | \\=15 | SPACE auto | ESC quit")

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

                if pygame.K_1 <= e.key <= pygame.K_9:
                    idx = e.key - pygame.K_1
                    auto = False
                elif e.key == pygame.K_0:
                    idx = 9
                    auto = False
                elif e.key == pygame.K_MINUS:
                    idx = 10
                    auto = False
                elif e.key == pygame.K_EQUALS:
                    idx = 11
                    auto = False
                elif e.key == pygame.K_LEFTBRACKET:
                    idx = 12
                    auto = False
                elif e.key == pygame.K_RIGHTBRACKET:
                    idx = 13
                    auto = False
                elif e.key == pygame.K_BACKSLASH:
                    idx = 14
                    auto = False

        if auto and (now - last) > 2.0:
            idx = (idx + 1) % len(EMOS)
            last = now

        render_face(screen, EMOS[idx], t)
        time.sleep(0.016)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
