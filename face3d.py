#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import pygame
from pygame import gfxdraw

# --------------------------
# AA drawing helpers
# --------------------------
def aa_line(surf, p1, p2, color, w=2):
    # draw several AA lines for a thin but smooth stroke
    x1, y1 = p1; x2, y2 = p2
    gfxdraw.line(surf, int(x1), int(y1), int(x2), int(y2), color)
    # pseudo thickness
    for i in range(1, w):
        gfxdraw.line(surf, int(x1), int(y1+i), int(x2), int(y2+i), color)
        gfxdraw.line(surf, int(x1), int(y1-i), int(x2), int(y2-i), color)

def aa_ellipse(surf, rect, color, filled=False):
    x, y, w, h = rect
    cx = x + w//2
    cy = y + h//2
    rx = w//2
    ry = h//2
    if filled:
        gfxdraw.filled_ellipse(surf, cx, cy, rx, ry, color)
    gfxdraw.aaellipse(surf, cx, cy, rx, ry, color)

def aa_circle(surf, center, r, color, filled=False):
    x, y = center
    if filled:
        gfxdraw.filled_circle(surf, int(x), int(y), int(r), color)
    gfxdraw.aacircle(surf, int(x), int(y), int(r), color)

def soft_glow_ellipse(surf, rect, base_alpha=30, layers=10):
    # white-only soft glow for subtle 3D-ish look
    x, y, w, h = rect
    for i in range(layers, 0, -1):
        pad = i * 2
        a = int(base_alpha * (i / layers))
        col = (255, 255, 255, a)
        aa_ellipse(surf, (x - pad, y - pad, w + pad*2, h + pad*2), col, filled=False)

# --------------------------
# Face drawing
# --------------------------
def blink_value(t, period=4.2):
    # returns 0..1 (0 open, 1 closed) with quick blink
    phase = (t % period) / period
    if phase < 0.03:
        return phase / 0.03
    if phase < 0.06:
        return 1 - (phase - 0.03) / 0.03
    return 0.0

def pupil_offset(t, amp):
    return (
        math.sin(t * 1.3) * amp,
        math.cos(t * 1.1) * amp * 0.7
    )

def mouth_wiggle(t, amp):
    return math.sin(t * 1.7) * amp

def draw_eye(surf, center, size, t, style="normal", blink=0.0):
    cx, cy = center
    ew, eh = size
    stroke = (255, 255, 255, 220)

    # eyelid blink -> squash height
    eh2 = max(2, int(eh * (1 - 0.85 * blink)))

    # eye outline (thin)
    rect = (int(cx - ew/2), int(cy - eh2/2), int(ew), int(eh2))
    soft_glow_ellipse(surf, rect, base_alpha=18, layers=8)
    aa_ellipse(surf, rect, stroke, filled=False)

    # pupil (only when open enough)
    if eh2 > eh * 0.25:
        ox, oy = pupil_offset(t, amp=ew * 0.10)
        # style tweaks
        if style == "side":
            ox *= 1.6
        if style == "up":
            oy -= ew * 0.06
        if style == "down":
            oy += ew * 0.06
        if style == "wink":
            # wink handled outside (blink bigger), keep small pupil
            ox *= 0.5; oy *= 0.5

        pr = max(2, int(min(ew, eh) * 0.12))
        aa_circle(surf, (cx + ox, cy + oy), pr, stroke, filled=True)

        # tiny highlight (3D feel) -> small alpha dot
        aa_circle(surf, (cx + ox - pr*0.35, cy + oy - pr*0.35), max(1, pr//3), (255,255,255,90), filled=True)

def draw_nose(surf, center, size, t, kind="round"):
    cx, cy = center
    stroke = (255, 255, 255, 220)
    wig = mouth_wiggle(t, amp=size * 0.04)

    if kind == "round":
        r = int(size * 0.18)
        aa_circle(surf, (cx, cy + wig), r, stroke, filled=True)
        aa_circle(surf, (cx - r*0.35, cy + wig - r*0.25), max(1, r//4), (255,255,255,70), filled=True)
    elif kind == "triangle":
        w = int(size * 0.40)
        h = int(size * 0.26)
        pts = [(cx, cy - h//2 + wig), (cx - w//2, cy + h//2 + wig), (cx + w//2, cy + h//2 + wig)]
        # filled triangle (white with a bit lower alpha) for softer look
        gfxdraw.filled_polygon(surf, [(int(x), int(y)) for x,y in pts], (255,255,255,200))
        gfxdraw.aapolygon(surf, [(int(x), int(y)) for x,y in pts], stroke)
    else:
        # tiny
        r = int(size * 0.12)
        aa_circle(surf, (cx, cy + wig), r, stroke, filled=True)

def draw_mouth(surf, center, width, t, mood="neutral"):
    cx, cy = center
    stroke = (255, 255, 255, 220)
    w = width
    wig = mouth_wiggle(t, amp=w * 0.02)

    if mood == "neutral":
        p1 = (cx - w*0.18, cy + wig)
        p2 = (cx + w*0.18, cy + wig)
        aa_line(surf, p1, p2, stroke, w=2)

    elif mood == "smile":
        # curve (polyline)
        pts = []
        for i in range(0, 21):
            u = i / 20
            x = cx - w*0.22 + u * w*0.44
            y = cy + w*0.05 + (u - 0.5)**2 * w*0.20 + wig
            pts.append((x, y))
        for i in range(len(pts)-1):
            aa_line(surf, pts[i], pts[i+1], stroke, w=2)

    elif mood == "sad":
        pts = []
        for i in range(0, 21):
            u = i / 20
            x = cx - w*0.22 + u * w*0.44
            y = cy - w*0.05 - (u - 0.5)**2 * w*0.20 + wig
            pts.append((x, y))
        for i in range(len(pts)-1):
            aa_line(surf, pts[i], pts[i+1], stroke, w=2)

    elif mood == "open":
        # small open mouth
        mw = int(w * 0.22)
        mh = int(w * 0.14 + abs(wig)*2)
        rect = (int(cx - mw/2), int(cy - mh/2), mw, mh)
        soft_glow_ellipse(surf, rect, base_alpha=14, layers=7)
        aa_ellipse(surf, rect, stroke, filled=False)

def draw_face(surf, W, H, t, idx):
    # black bg already
    cx, cy = W//2, H//2
    scale = min(W, H)

    # layout
    eye_y = cy - int(scale * 0.10)
    eye_dx = int(scale * 0.18)
    eye_size = (int(scale*0.12), int(scale*0.07))

    nose_y = cy + int(scale * 0.03)
    mouth_y = cy + int(scale * 0.18)

    b = blink_value(t, period=4.0 + (idx % 3) * 0.6)

    # per-face styles roughly matching your 3x3 idea
    if idx == 1:   # cute neutral
        draw_eye(surf, (cx-eye_dx, eye_y), eye_size, t, "normal", blink=b*0.3)
        draw_eye(surf, (cx+eye_dx, eye_y), eye_size, t, "normal", blink=b*0.3)
        draw_nose(surf, (cx, nose_y), scale*0.30, t, "round")
        draw_mouth(surf, (cx, mouth_y), scale*0.45, t, "smile")

    elif idx == 2: # sad droopy
        draw_eye(surf, (cx-eye_dx, eye_y), (eye_size[0], int(eye_size[1]*0.75)), t, "down", blink=b*0.25)
        draw_eye(surf, (cx+eye_dx, eye_y), (eye_size[0], int(eye_size[1]*0.75)), t, "down", blink=b*0.25)
        draw_nose(surf, (cx, nose_y), scale*0.34, t, "triangle")
        draw_mouth(surf, (cx, mouth_y), scale*0.45, t, "sad")

    elif idx == 3: # side look
        draw_eye(surf, (cx-eye_dx, eye_y), eye_size, t, "side", blink=b*0.25)
        draw_eye(surf, (cx+eye_dx, eye_y), eye_size, t, "side", blink=b*0.25)
        draw_nose(surf, (cx, nose_y), scale*0.30, t, "round")
        draw_mouth(surf, (cx, mouth_y), scale*0.42, t, "neutral")

    elif idx == 4: # suspicious (one eye smaller)
        draw_eye(surf, (cx-eye_dx, eye_y), (int(eye_size[0]*0.75), eye_size[1]), t, "normal", blink=b*0.15)
        draw_eye(surf, (cx+eye_dx, eye_y), (int(eye_size[0]*1.05), eye_size[1]), t, "up", blink=b*0.15)
        draw_nose(surf, (cx, nose_y), scale*0.28, t, "tiny")
        draw_mouth(surf, (cx, mouth_y), scale*0.44, t, "neutral")

    elif idx == 5: # shocked / worried
        draw_eye(surf, (cx-eye_dx, eye_y), (eye_size[0], int(eye_size[1]*1.15)), t, "up", blink=b*0.05)
        draw_eye(surf, (cx+eye_dx, eye_y), (eye_size[0], int(eye_size[1]*1.15)), t, "up", blink=b*0.05)
        draw_nose(surf, (cx, nose_y), scale*0.36, t, "triangle")
        draw_mouth(surf, (cx, mouth_y), scale*0.46, t, "open")

    elif idx == 6: # friendly smile with smaller pupils
        draw_eye(surf, (cx-eye_dx, eye_y), (int(eye_size[0]*0.9), eye_size[1]), t, "normal", blink=b*0.35)
        draw_eye(surf, (cx+eye_dx, eye_y), (int(eye_size[0]*0.9), eye_size[1]), t, "normal", blink=b*0.35)
        draw_nose(surf, (cx, nose_y), scale*0.33, t, "round")
        draw_mouth(surf, (cx, mouth_y), scale*0.48, t, "smile")

    elif idx == 7: # wink
        # left wink: force blink high
        draw_eye(surf, (cx-eye_dx, eye_y), eye_size, t, "wink", blink=0.9)
        draw_eye(surf, (cx+eye_dx, eye_y), eye_size, t, "normal", blink=b*0.25)
        draw_nose(surf, (cx, nose_y), scale*0.26, t, "tiny")
        draw_mouth(surf, (cx, mouth_y), scale*0.44, t, "smile")

    elif idx == 8: # big nose + neutral mouth
        draw_eye(surf, (cx-eye_dx, eye_y), (int(eye_size[0]*0.85), eye_size[1]), t, "normal", blink=b*0.2)
        draw_eye(surf, (cx+eye_dx, eye_y), (int(eye_size[0]*0.85), eye_size[1]), t, "normal", blink=b*0.2)
        draw_nose(surf, (cx, nose_y), scale*0.40, t, "triangle")
        draw_mouth(surf, (cx, mouth_y), scale*0.40, t, "neutral")

    elif idx == 9: # laughing / open smile
        draw_eye(surf, (cx-eye_dx, eye_y), (int(eye_size[0]*0.95), int(eye_size[1]*0.65)), t, "normal", blink=b*0.45)
        draw_eye(surf, (cx+eye_dx, eye_y), (int(eye_size[0]*0.95), int(eye_size[1]*0.65)), t, "normal", blink=b*0.45)
        draw_nose(surf, (cx, nose_y), scale*0.34, t, "round")
        draw_mouth(surf, (cx, mouth_y), scale*0.52, t, "open")

# --------------------------
# Main
# --------------------------
def main():
    pygame.init()

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H = screen.get_size()
    clock = pygame.time.Clock()

    idx = 1
    t = 0.0
    running = True

    # pre-create transparent layer for alpha drawings
    layer = pygame.Surface((W, H), pygame.SRCALPHA)

    while running:
        dt = clock.tick(60) / 1000.0
        t += dt

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                if pygame.K_1 <= e.key <= pygame.K_9:
                    idx = e.key - pygame.K_0

        # background
        screen.fill((0, 0, 0))
        layer.fill((0, 0, 0, 0))

        draw_face(layer, W, H, t, idx)

        # blit alpha layer
        screen.blit(layer, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
