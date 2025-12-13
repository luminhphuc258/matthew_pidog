#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, math
import pygame

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ===== STYLE =====
BG = (0, 0, 0)

# Orange neon
ORANGE  = (255, 170, 60)   # core
ORANGE2 = (255, 210, 130)  # glow/highlight
HILITE  = (255, 245, 220)

# Thin strokes
THICK_LINE  = 4
THICK_EYE   = 4
THICK_MOUTH = 5

# ===== KEY MAPPING (updated) =====
KEY_TO_EMO = {
    "1": "love_eyes",
    "2": "music",
    "4": "what_is_it",
    "6": "suprise",
    "7": "sleep",
    "9": "sad",
    "0": "angry",
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_glow_line(screen, p1, p2, width):
    pygame.draw.line(screen, ORANGE2, p1, p2, max(1, width + 6))
    pygame.draw.line(screen, ORANGE,  p1, p2, max(1, width))

def draw_glow_arc(screen, rect, start, end, width):
    pygame.draw.arc(screen, ORANGE2, rect, start, end, max(1, width + 6))
    pygame.draw.arc(screen, ORANGE,  rect, start, end, max(1, width))

def draw_glow_rect_outline(screen, rect, radius, width):
    pygame.draw.rect(screen, ORANGE2, rect, max(1, width + 6), border_radius=radius)
    pygame.draw.rect(screen, ORANGE,  rect, max(1, width),     border_radius=radius)

def draw_glow_circle(screen, center, r, width):
    if width <= 0:
        pygame.draw.circle(screen, ORANGE2, center, r + 6)
        pygame.draw.circle(screen, ORANGE,  center, r)
        return
    pygame.draw.circle(screen, ORANGE2, center, r, max(1, width + 6))
    pygame.draw.circle(screen, ORANGE,  center, r, max(1, width))

def eye_open(screen, cx, cy, w, h, pupil_dx=0, pupil_dy=0, blink=0.0):
    hh = max(8, int(h * (1.0 - 0.90 * blink)))
    rect = pygame.Rect(cx - w//2, cy - hh//2, w, hh)
    radius = max(10, hh//2)

    draw_glow_rect_outline(screen, rect, radius=radius, width=THICK_EYE)

    pr = max(8, int(min(w, hh) * 0.14))
    px = clamp(cx + pupil_dx, rect.left + pr + 8, rect.right - pr - 8)
    py = clamp(cy + pupil_dy, rect.top  + pr + 8, rect.bottom - pr - 8)

    pygame.draw.circle(screen, ORANGE2, (px, py), pr + 4)
    pygame.draw.circle(screen, ORANGE,  (px, py), pr)
    pygame.draw.circle(screen, HILITE, (px - pr//3, py - pr//3), max(2, pr//3))

def eye_happy_arc(screen, cx, cy, w, h):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    draw_glow_arc(screen, rect, math.pi, 2*math.pi, THICK_LINE)

def mouth_smile(screen, cx, cy, w, h, open_amt=0.0, wob=0):
    rect = pygame.Rect(cx - w//2, cy - h//2 + wob, w, h)
    draw_glow_arc(screen, rect, 0, math.pi, THICK_MOUTH)

    if open_amt > 0:
        ow = int(w * 0.30)
        oh = int(h * 0.55 * open_amt)
        if oh > 8:
            r = max(10, oh//2)
            rr = pygame.Rect(cx - ow//2, cy + int(h*0.10) + wob, ow, oh)
            draw_glow_rect_outline(screen, rr, radius=r, width=THICK_LINE)

def mouth_open(screen, cx, cy, w, h, wob=0):
    rect = pygame.Rect(cx - w//2, cy - h//2 + wob, w, h)
    r = max(12, h//2)
    draw_glow_rect_outline(screen, rect, radius=r, width=THICK_MOUTH)

def mouth_flat(screen, cx, cy, w, wob=0):
    draw_glow_line(screen, (cx - w//2, cy + wob), (cx + w//2, cy + wob), THICK_MOUTH)

def q_tears(screen, x, y, t):
    bob = int(6 * math.sin(t * 2.7))
    rect = pygame.Rect(x - 9, y + 18 + bob, 18, 36)
    pygame.draw.ellipse(screen, ORANGE2, rect, 6)
    pygame.draw.ellipse(screen, ORANGE,  rect, 3)

def ex_mark(screen, cx, cy, t):
    bounce = int(8 * abs(math.sin(t*3.0)))
    top = (cx, cy - 150 - bounce)
    mid = (cx, cy - 98  - bounce)
    draw_glow_line(screen, top, mid, THICK_LINE)
    draw_glow_circle(screen, (cx, cy - 70 - bounce), 6, 0)

def zzz(screen, cx, cy, t):
    for i in range(3):
        yy = cy - 165 - i*44 - int(8*math.sin(t*2 + i))
        xx = cx + 165 + i*28
        draw_glow_line(screen, (xx-16, yy-12), (xx+16, yy-12), THICK_LINE)
        draw_glow_line(screen, (xx+16, yy-12), (xx-16, yy+12), THICK_LINE)
        draw_glow_line(screen, (xx-16, yy+12), (xx+16, yy+12), THICK_LINE)

# ===== NEW: HEART EYES =====
def draw_heart(screen, x, y, s, t):
    # heart = 2 circles + triangle/polygon; subtle pulse
    pulse = 1.0 + 0.08 * math.sin(t * 3.6)
    s = int(s * pulse)
    r = max(10, int(s * 0.28))
    # glow layer (bigger)
    pygame.draw.circle(screen, ORANGE2, (x - r, y), r + 5)
    pygame.draw.circle(screen, ORANGE2, (x + r, y), r + 5)
    pygame.draw.polygon(screen, ORANGE2, [(x - 2*r - 5, y), (x + 2*r + 5, y), (x, y + int(2.2*r) + 5)])

    # core
    pygame.draw.circle(screen, ORANGE, (x - r, y), r)
    pygame.draw.circle(screen, ORANGE, (x + r, y), r)
    pygame.draw.polygon(screen, ORANGE, [(x - 2*r, y), (x + 2*r, y), (x, y + int(2.2*r))])

    # highlight
    pygame.draw.circle(screen, HILITE, (x - r - max(2, r//3), y - max(2, r//3)), max(2, r//3))

# ===== NEW: MUSIC NOTE EYES (like your icon) =====
def draw_music_note(screen, x, y, s, t, flip=False):
    """
    Draw a simple quaver/eighth-note:
    - head: filled circle
    - stem: vertical line
    - flag: curved-ish polyline
    """
    # gentle bob
    bob = int(3 * math.sin(t * 2.8 + (0.8 if flip else 0.0)))
    s = int(s * (1.0 + 0.05 * math.sin(t * 3.0)))
    head_r = max(10, int(s * 0.18))

    # head position
    hx = x
    hy = y + bob + int(s * 0.18)

    # stem
    stem_h = int(s * 0.70)
    stem_w = THICK_LINE
    stem_dir = -1 if not flip else 1  # flip just changes flag side
    stem_x = hx + int(head_r * 0.9)
    stem_y1 = hy - int(head_r * 0.2)
    stem_y2 = stem_y1 - stem_h

    # glow head
    pygame.draw.circle(screen, ORANGE2, (hx, hy), head_r + 5)
    pygame.draw.circle(screen, ORANGE,  (hx, hy), head_r)

    # glow stem
    draw_glow_line(screen, (stem_x, stem_y1), (stem_x, stem_y2), stem_w)

    # flag (polyline)
    fx0, fy0 = stem_x, stem_y2
    # make a small "C" shape using segments
    flag = [
        (fx0, fy0),
        (fx0 + stem_dir*int(s*0.20), fy0 + int(s*0.08)),
        (fx0 + stem_dir*int(s*0.26), fy0 + int(s*0.20)),
        (fx0 + stem_dir*int(s*0.10), fy0 + int(s*0.26)),
    ]
    for i in range(len(flag)-1):
        draw_glow_line(screen, flag[i], flag[i+1], THICK_LINE)

    # highlight
    pygame.draw.circle(screen, HILITE, (hx - head_r//3, hy - head_r//3), max(2, head_r//3))

def render_face(screen, emo, t):
    W, H = screen.get_size()
    screen.fill(BG)

    cx, cy = W//2, H//2

    # Fullscreen layout
    eye_y   = cy - int(H * 0.14)
    mouth_y = cy + int(H * 0.18)

    eye_w = int(W * 0.32)
    eye_h = int(H * 0.22)
    gap   = int(W * 0.18)

    lx = cx - gap
    rx = cx + gap

    mouth_w = int(W * 0.56)
    mouth_h = int(H * 0.28)

    # subtle motion
    phase = (t % 3.3)
    blink = 0.0
    if phase < 0.10:
        blink = 1.0 - phase/0.10
    elif phase < 0.20:
        blink = (phase-0.10)/0.10
    blink = clamp(blink, 0.0, 1.0)

    pdx = int(math.sin(t*1.7) * (W * 0.012))
    pdy = int(math.sin(t*1.2 + 1.1) * (H * 0.010))
    wob = int(math.sin(t*2.4) * (H * 0.010))

    # ===== LOVE (heart eyes + normal mouth) =====
    if emo == "love_eyes":
        heart_size = int(min(W, H) * 0.22)
        draw_heart(screen, lx, eye_y, heart_size, t)
        draw_heart(screen, rx, eye_y, heart_size, t + 0.25)
        # normal mouth (small smile)
        mouth_smile(screen, cx, mouth_y, int(mouth_w*0.60), int(mouth_h*0.70),
                    open_amt=0.10 + 0.05*math.sin(t*3.6), wob=wob)

    # ===== MUSIC (note eyes + normal mouth) =====
    elif emo == "music":
        note_size = int(min(W, H) * 0.28)
        draw_music_note(screen, lx, eye_y - int(H*0.02), note_size, t, flip=False)
        draw_music_note(screen, rx, eye_y - int(H*0.02), note_size, t + 0.3, flip=True)
        mouth_smile(screen, cx, mouth_y, int(mouth_w*0.58), int(mouth_h*0.70),
                    open_amt=0.08 + 0.05*math.sin(t*3.2), wob=wob)

    # ===== what is it =====
    elif emo == "what_is_it":
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy, blink*0.25)
        eye_open(screen, rx, eye_y, eye_w, eye_h,  pdx, -pdy, blink*0.25)
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.52), wob=wob)

    # ===== suprise =====
    elif emo == "suprise":
        eye_open(screen, lx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        eye_open(screen, rx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        mouth_open(screen, cx, mouth_y, int(mouth_w*0.20), int(mouth_h*0.24), wob=wob)
        ex_mark(screen, cx, cy, t)

    # ===== sleep =====
    elif emo == "sleep":
        eye_happy_arc(screen, lx, eye_y + 14, int(eye_w*0.88), int(eye_h*0.62))
        eye_happy_arc(screen, rx, eye_y + 14, int(eye_w*0.88), int(eye_h*0.62))
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.32), wob=wob)
        zzz(screen, cx, cy, t)

    # ===== sad =====
    elif emo == "sad":
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy + 10, blink*0.15)
        eye_open(screen, rx, eye_y, eye_w, eye_h,  pdx, pdy + 10, blink*0.15)
        rect = pygame.Rect(cx - int(mouth_w*0.65)//2, mouth_y - int(mouth_h*0.55)//2 + wob,
                           int(mouth_w*0.65), int(mouth_h*0.55))
        draw_glow_arc(screen, rect, math.pi, 2*math.pi, THICK_MOUTH)
        q_tears(screen, lx + int(eye_w*0.22), eye_y + int(eye_h*0.06), t)

    # ===== angry =====
    elif emo == "angry":
        jitter = int(math.sin(t*10.0) * 6)
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx + jitter, pdy, blink*0.05)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx - jitter, pdy, blink*0.05)

        draw_glow_line(screen,
                       (lx - int(eye_w*0.55), eye_y - int(eye_h*0.52)),
                       (lx + int(eye_w*0.55), eye_y - int(eye_h*0.18)), THICK_LINE)
        draw_glow_line(screen,
                       (rx - int(eye_w*0.55), eye_y - int(eye_h*0.18)),
                       (rx + int(eye_w*0.55), eye_y - int(eye_h*0.52)), THICK_LINE)

        amp = int(H * 0.010)
        pts = []
        w = int(mouth_w * 0.58)
        for i in range(54):
            x = cx - w//2 + (i/53.0)*w
            y = mouth_y + wob + math.sin(t*7.0 + i*0.30) * amp
            pts.append((x, y))
        pygame.draw.lines(screen, ORANGE2, False, pts, THICK_MOUTH + 4)
        pygame.draw.lines(screen, ORANGE,  False, pts, THICK_MOUTH)

    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    emo = "music"
    auto = False
    last = time.time()

    order = ["love_eyes","music","what_is_it","suprise","sleep","sad","angry"]
    idx = 0

    print("Keys: 1=love_eyes 2=music 4=what_is_it 6=suprise 7=sleep 9=sad 0=angry | SPACE auto | ESC quit")

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

                if e.unicode in KEY_TO_EMO:
                    emo = KEY_TO_EMO[e.unicode]
                    auto = False

        if auto and (now - last) > 2.0:
            emo = order[idx % len(order)]
            idx += 1
            last = now

        render_face(screen, emo, t)
        time.sleep(0.016)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
