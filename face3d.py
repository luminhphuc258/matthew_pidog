#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, math
import pygame

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ===== STYLE =====
BG = (0, 0, 0)

# Orange neon like your sample
ORANGE  = (255, 170, 60)   # core
ORANGE2 = (255, 210, 130)  # glow/highlight

# Thin strokes (mỏng thôi)
THICK_LINE  = 4
THICK_EYE   = 4
THICK_MOUTH = 5

# Key mapping (as you requested)
KEY_TO_EMO = {
    "1": "love",
    "2": "smile",
    "3": "laugh",
    "4": "what_is_it",
    "5": "question",
    "6": "suprise",
    "7": "sleep",
    "9": "sad",
    "0": "angry",
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_glow_line(screen, p1, p2, width):
    # subtle glow (3D-ish) using lighter orange behind
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
    # eyelid effect by shrinking height
    hh = max(8, int(h * (1.0 - 0.90 * blink)))
    rect = pygame.Rect(cx - w//2, cy - hh//2, w, hh)
    radius = max(10, hh//2)

    draw_glow_rect_outline(screen, rect, radius=radius, width=THICK_EYE)

    # pupil (soft 3D: glow + highlight)
    pr = max(8, int(min(w, hh) * 0.14))
    px = clamp(cx + pupil_dx, rect.left + pr + 8, rect.right - pr - 8)
    py = clamp(cy + pupil_dy, rect.top  + pr + 8, rect.bottom - pr - 8)

    pygame.draw.circle(screen, ORANGE2, (px, py), pr + 4)
    pygame.draw.circle(screen, ORANGE,  (px, py), pr)

    # tiny highlight
    pygame.draw.circle(screen, (255, 245, 220), (px - pr//3, py - pr//3), max(2, pr//3))

def eye_happy_arc(screen, cx, cy, w, h):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    draw_glow_arc(screen, rect, math.pi, 2*math.pi, THICK_LINE)

def eye_dot(screen, cx, cy, r, t):
    # dot eye with gentle floating highlight
    bob = int(2 * math.sin(t * 2.4))
    pygame.draw.circle(screen, ORANGE2, (cx, cy + bob), r + 4)
    pygame.draw.circle(screen, ORANGE,  (cx, cy + bob), r)
    pygame.draw.circle(screen, (255,245,220), (cx - r//3, cy - r//3 + bob), max(2, r//3))

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

def mouth_laugh(screen, cx, cy, w, h, t):
    # wide smile + more open + bounce
    bounce = int(6 * abs(math.sin(t * 3.2)))
    mouth_smile(screen, cx, cy + bounce, w, h, open_amt=0.55 + 0.15*math.sin(t*4.5), wob=0)

def draw_heart_outline(screen, x, y, s, t):
    # outline heart (thin) with gentle pulse (matches your "love" icon)
    pulse = 1.0 + 0.06 * math.sin(t * 3.0)
    s = int(s * pulse)
    # param heart curve -> polyline
    pts = []
    for i in range(0, 181, 6):
        a = math.radians(i)
        px = 16 * (math.sin(a) ** 3)
        py = 13*math.cos(a) - 5*math.cos(2*a) - 2*math.cos(3*a) - math.cos(4*a)
        pts.append((x + int(px * s/18), y - int(py * s/18)))
    # glow outline
    pygame.draw.lines(screen, ORANGE2, False, pts, THICK_LINE + 4)
    pygame.draw.lines(screen, ORANGE,  False, pts, THICK_LINE)

def q_mark(screen, cx, cy, t):
    wob = int(6 * math.sin(t * 2.8))
    rect = pygame.Rect(cx - 28, cy - 165 + wob, 56, 68)
    draw_glow_arc(screen, rect, math.pi*1.1, math.pi*2.15, THICK_LINE)
    draw_glow_line(screen, (cx + 24, cy - 108 + wob), (cx + 24, cy - 92 + wob), THICK_LINE)
    draw_glow_circle(screen, (cx + 24, cy - 70 + wob), 6, 0)

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

def tears(screen, x, y, t):
    bob = int(6 * math.sin(t * 2.7))
    rect = pygame.Rect(x - 9, y + 18 + bob, 18, 36)
    pygame.draw.ellipse(screen, ORANGE2, rect, 6)
    pygame.draw.ellipse(screen, ORANGE,  rect, 3)

def render_face(screen, emo, t):
    W, H = screen.get_size()
    screen.fill(BG)

    cx, cy = W//2, H//2

    # ===== Fullscreen layout (big features) =====
    eye_y   = cy - int(H * 0.14)
    mouth_y = cy + int(H * 0.18)

    eye_w = int(W * 0.32)
    eye_h = int(H * 0.22)
    gap   = int(W * 0.18)

    lx = cx - gap
    rx = cx + gap

    mouth_w = int(W * 0.56)
    mouth_h = int(H * 0.28)

    # ===== subtle motion =====
    # blink
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

    # ================== NEW 3 faces (love/smile/laugh) ==================
    if emo == "love":
        # heart outline centered (big) + small smile; eyes are minimal like icon style
        draw_heart_outline(screen, cx, cy - int(H*0.10), int(min(W,H)*0.40), t)
        # tiny dot eyes
        eye_dot(screen, lx, eye_y, int(min(W,H)*0.028), t)
        eye_dot(screen, rx, eye_y, int(min(W,H)*0.028), t)
        mouth_smile(screen, cx, mouth_y, int(mouth_w*0.55), int(mouth_h*0.70), open_amt=0.10, wob=wob)

    elif emo == "smile":
        # icon-like: curved closed eyes + gentle smile
        eye_happy_arc(screen, lx, eye_y, int(eye_w*0.90), int(eye_h*0.70))
        eye_happy_arc(screen, rx, eye_y, int(eye_w*0.90), int(eye_h*0.70))
        mouth_smile(screen, cx, mouth_y, int(mouth_w*0.68), int(mouth_h*0.80),
                    open_amt=0.10 + 0.05*math.sin(t*3.8), wob=wob)

    elif emo == "laugh":
        # bigger smile + open mouth bounce
        eye_happy_arc(screen, lx, eye_y, int(eye_w*0.92), int(eye_h*0.70))
        eye_happy_arc(screen, rx, eye_y, int(eye_w*0.92), int(eye_h*0.70))
        mouth_laugh(screen, cx, mouth_y, int(mouth_w*0.78), int(mouth_h*0.90), t)

    # ================== Remapped faces you requested ==================
    elif emo == "what_is_it":
        # curious: eyes look opposite directions + flat mouth
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy, blink*0.25)
        eye_open(screen, rx, eye_y, eye_w, eye_h,  pdx, -pdy, blink*0.25)
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.52), wob=wob)

    elif emo == "question":
        # open eyes + small mouth + question mark
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx, pdy, blink*0.20)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx, pdy, blink*0.20)
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.40), wob=wob)
        q_mark(screen, cx, cy, t)

    elif emo == "suprise":
        # wide open eyes + small O mouth + exclamation
        eye_open(screen, lx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        eye_open(screen, rx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        mouth_open(screen, cx, mouth_y, int(mouth_w*0.20), int(mouth_h*0.24), wob=wob)
        ex_mark(screen, cx, cy, t)

    elif emo == "sleep":
        # closed eyes + flat mouth + ZZZ
        eye_happy_arc(screen, lx, eye_y + 14, int(eye_w*0.88), int(eye_h*0.62))
        eye_happy_arc(screen, rx, eye_y + 14, int(eye_w*0.88), int(eye_h*0.62))
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.32), wob=wob)
        zzz(screen, cx, cy, t)

    elif emo == "sad":
        # droopy pupils + sad mouth + tear
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy + 10, blink*0.15)
        eye_open(screen, rx, eye_y, eye_w, eye_h,  pdx, pdy + 10, blink*0.15)
        # sad arc mouth (upside down smile)
        rect = pygame.Rect(cx - int(mouth_w*0.65)//2, mouth_y - int(mouth_h*0.55)//2 + wob,
                           int(mouth_w*0.65), int(mouth_h*0.55))
        draw_glow_arc(screen, rect, math.pi, 2*math.pi, THICK_MOUTH)
        tears(screen, lx + int(eye_w*0.22), eye_y + int(eye_h*0.06), t)

    elif emo == "angry":
        # jitter pupils + angry brows + wavy mouth
        jitter = int(math.sin(t*10.0) * 6)
        eye_open(screen, lx, eye_y, eye_w, eye_h, pdx + jitter, pdy, blink*0.05)
        eye_open(screen, rx, eye_y, eye_w, eye_h, pdx - jitter, pdy, blink*0.05)

        # brows
        draw_glow_line(screen,
                       (lx - int(eye_w*0.55), eye_y - int(eye_h*0.52)),
                       (lx + int(eye_w*0.55), eye_y - int(eye_h*0.18)), THICK_LINE)
        draw_glow_line(screen,
                       (rx - int(eye_w*0.55), eye_y - int(eye_h*0.18)),
                       (rx + int(eye_w*0.55), eye_y - int(eye_h*0.52)), THICK_LINE)

        # wavy mouth
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

    emo = "smile"
    auto = False
    last = time.time()

    print("Keys: 1=love 2=smile 3=laugh 4=what_is_it 5=question 6=suprise 7=sleep 9=sad 0=angry | SPACE auto | ESC quit")

    order = ["love","smile","laugh","what_is_it","question","suprise","sleep","sad","angry"]
    idx = 1

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

                # digit keys mapping
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
