#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, math, threading
import pygame

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# ===== STYLE =====
BG = (0, 0, 0)

ORANGE  = (255, 170, 60)   # core
ORANGE2 = (255, 210, 130)  # glow/highlight
HILITE  = (255, 245, 220)

THICK_LINE  = 3
THICK_EYE   = 3
THICK_MOUTH = 4


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_glow_line(screen, p1, p2, width):
    pygame.draw.line(screen, ORANGE2, p1, p2, max(1, width + 5))
    pygame.draw.line(screen, ORANGE,  p1, p2, max(1, width))

def draw_glow_arc(screen, rect, start, end, width):
    pygame.draw.arc(screen, ORANGE2, rect, start, end, max(1, width + 5))
    pygame.draw.arc(screen, ORANGE,  rect, start, end, max(1, width))

def draw_glow_rect_outline(screen, rect, radius, width):
    pygame.draw.rect(screen, ORANGE2, rect, max(1, width + 5), border_radius=radius)
    pygame.draw.rect(screen, ORANGE,  rect, max(1, width),     border_radius=radius)

def draw_glow_circle(screen, center, r, width):
    if width <= 0:
        pygame.draw.circle(screen, ORANGE2, center, r + 5)
        pygame.draw.circle(screen, ORANGE,  center, r)
        return
    pygame.draw.circle(screen, ORANGE2, center, r, max(1, width + 5))
    pygame.draw.circle(screen, ORANGE,  center, r, max(1, width))

# --------------------------
# Eyes / Mouth
# --------------------------
def eye_open(screen, cx, cy, w, h, pupil_dx=0, pupil_dy=0, blink=0.0):
    hh = max(8, int(h * (1.0 - 0.90 * blink)))
    rect = pygame.Rect(cx - w//2, cy - hh//2, w, hh)
    radius = max(10, hh//2)

    draw_glow_rect_outline(screen, rect, radius=radius, width=THICK_EYE)

    pr = max(7, int(min(w, hh) * 0.13))
    px = clamp(cx + pupil_dx, rect.left + pr + 8, rect.right - pr - 8)
    py = clamp(cy + pupil_dy, rect.top  + pr + 8, rect.bottom - pr - 8)

    pygame.draw.circle(screen, ORANGE2, (px, py), pr + 3)
    pygame.draw.circle(screen, ORANGE,  (px, py), pr)
    pygame.draw.circle(screen, HILITE,  (px - pr//3, py - pr//3), max(2, pr//3))

def eye_happy_arc(screen, cx, cy, w, h):
    rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
    draw_glow_arc(screen, rect, math.pi, 2*math.pi, THICK_LINE)

def mouth_smile_arc_only(screen, cx, cy, w, h, t, wob=0):
    wob2 = wob + int(2 * math.sin(t * 2.8))
    rect = pygame.Rect(cx - w//2, cy - h//2 + wob2, w, h)
    draw_glow_arc(screen, rect, 0, math.pi, THICK_MOUTH)

def mouth_open(screen, cx, cy, w, h, wob=0):
    rect = pygame.Rect(cx - w//2, cy - h//2 + wob, w, h)
    r = max(12, h//2)
    draw_glow_rect_outline(screen, rect, radius=r, width=THICK_MOUTH)

def mouth_flat(screen, cx, cy, w, wob=0):
    draw_glow_line(screen, (cx - w//2, cy + wob), (cx + w//2, cy + wob), THICK_MOUTH)

# --------------------------
# Symbols
# --------------------------
def ex_mark(screen, cx, cy, t):
    bounce = int(7 * abs(math.sin(t * 3.0)))
    top = (cx, cy - 150 - bounce)
    mid = (cx, cy - 100 - bounce)
    draw_glow_line(screen, top, mid, THICK_LINE)
    draw_glow_circle(screen, (cx, cy - 70 - bounce), 6, 0)

def zzz(screen, cx, cy, t):
    for i in range(3):
        yy = cy - 165 - i*44 - int(7*math.sin(t*2 + i))
        xx = cx + 165 + i*28
        draw_glow_line(screen, (xx-16, yy-12), (xx+16, yy-12), THICK_LINE)
        draw_glow_line(screen, (xx+16, yy-12), (xx-16, yy+12), THICK_LINE)
        draw_glow_line(screen, (xx-16, yy+12), (xx+16, yy+12), THICK_LINE)

def tears(screen, x, y, t):
    bob = int(6 * math.sin(t * 2.7))
    rect = pygame.Rect(x - 9, y + 18 + bob, 18, 36)
    pygame.draw.ellipse(screen, ORANGE2, rect, 5)
    pygame.draw.ellipse(screen, ORANGE,  rect, 3)

# --------------------------
# Heart & Music eyes
# --------------------------
def draw_heart(screen, x, y, s, t):
    pulse = 1.0 + 0.08 * math.sin(t * 3.6)
    s = int(s * pulse)
    r = max(10, int(s * 0.28))

    pygame.draw.circle(screen, ORANGE2, (x - r, y), r + 4)
    pygame.draw.circle(screen, ORANGE2, (x + r, y), r + 4)
    pygame.draw.polygon(screen, ORANGE2, [(x - 2*r - 4, y), (x + 2*r + 4, y), (x, y + int(2.2*r) + 4)])

    pygame.draw.circle(screen, ORANGE, (x - r, y), r)
    pygame.draw.circle(screen, ORANGE, (x + r, y), r)
    pygame.draw.polygon(screen, ORANGE, [(x - 2*r, y), (x + 2*r, y), (x, y + int(2.2*r))])

    pygame.draw.circle(screen, HILITE, (x - r - max(2, r//3), y - max(2, r//3)), max(2, r//3))

def draw_music_note(screen, x, y, s, t, flip=False):
    bob = int(3 * math.sin(t * 2.8 + (0.8 if flip else 0.0)))
    s = int(s * (1.0 + 0.05 * math.sin(t * 3.0)))
    head_r = max(10, int(s * 0.18))

    hx = x
    hy = y + bob + int(s * 0.18)

    stem_h = int(s * 0.70)
    stem_dir = -1 if not flip else 1

    stem_x = hx + int(head_r * 0.9)
    stem_y1 = hy - int(head_r * 0.2)
    stem_y2 = stem_y1 - stem_h

    pygame.draw.circle(screen, ORANGE2, (hx, hy), head_r + 4)
    pygame.draw.circle(screen, ORANGE,  (hx, hy), head_r)

    draw_glow_line(screen, (stem_x, stem_y1), (stem_x, stem_y2), THICK_LINE)

    fx0, fy0 = stem_x, stem_y2
    flag = [
        (fx0, fy0),
        (fx0 + stem_dir*int(s*0.20), fy0 + int(s*0.08)),
        (fx0 + stem_dir*int(s*0.26), fy0 + int(s*0.20)),
        (fx0 + stem_dir*int(s*0.10), fy0 + int(s*0.26)),
    ]
    for i in range(len(flag)-1):
        draw_glow_line(screen, flag[i], flag[i+1], THICK_LINE)

    pygame.draw.circle(screen, HILITE, (hx - head_r//3, hy - head_r//3), max(2, head_r//3))


def render_face(screen, emo, t):
    W, H = screen.get_size()
    screen.fill(BG)

    cx, cy = W//2, H//2

    eye_y   = cy - int(H * 0.14)
    mouth_y = cy + int(H * 0.18)

    eye_w = int(W * 0.32)
    eye_h = int(H * 0.22)
    gap   = int(W * 0.18)

    lx = cx - gap
    rx = cx + gap

    mouth_w = int(W * 0.56)
    mouth_h = int(H * 0.28)

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

    if emo == "love_eyes":
        heart_size = int(min(W, H) * 0.22)
        draw_heart(screen, lx, eye_y, heart_size, t)
        draw_heart(screen, rx, eye_y, heart_size, t + 0.25)

        mouth_smile_arc_only(screen, cx, mouth_y,
                             int(mouth_w * 0.36),
                             int(mouth_h * 0.22),
                             t, wob=wob)

    elif emo == "music":
        note_size = int(min(W, H) * 0.28)
        draw_music_note(screen, lx, eye_y - int(H*0.02), note_size, t, flip=False)
        draw_music_note(screen, rx, eye_y - int(H*0.02), note_size, t + 0.3, flip=True)

        mouth_smile_arc_only(screen, cx, mouth_y,
                             int(mouth_w * 0.36),
                             int(mouth_h * 0.22),
                             t, wob=wob)

    elif emo == "what_is_it":
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy, blink*0.25)
        eye_open(screen, rx, eye_y, eye_w, eye_h,  pdx, -pdy, blink*0.25)
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.52), wob=wob)

    elif emo == "suprise":
        eye_open(screen, lx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        eye_open(screen, rx, eye_y, eye_w, eye_h, 0, 0, 0.0)
        mouth_open(screen, cx, mouth_y, int(mouth_w*0.20), int(mouth_h*0.24), wob=wob)
        ex_mark(screen, cx, cy, t)

    elif emo == "sleep":
        eye_happy_arc(screen, lx, eye_y + 14, int(eye_w*0.88), int(eye_h*0.62))
        eye_happy_arc(screen, rx, eye_y + 14, int(eye_w*0.88), int(eye_h*0.62))
        mouth_flat(screen, cx, mouth_y, int(mouth_w*0.32), wob=wob)
        zzz(screen, cx, cy, t)

    elif emo == "sad":
        eye_open(screen, lx, eye_y, eye_w, eye_h, -pdx, pdy + 10, blink*0.15)
        eye_open(screen, rx, eye_y, eye_w, eye_h,  pdx, pdy + 10, blink*0.15)

        rect = pygame.Rect(
            cx - int(mouth_w*0.36)//2,
            mouth_y - int(mouth_h*0.22)//2 + wob,
            int(mouth_w*0.36),
            int(mouth_h*0.22)
        )
        draw_glow_arc(screen, rect, math.pi, 2*math.pi, THICK_MOUTH)
        tears(screen, lx + int(eye_w*0.22), eye_y + int(eye_h*0.06), t)

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
        pygame.draw.lines(screen, ORANGE2, False, pts, THICK_MOUTH + 3)
        pygame.draw.lines(screen, ORANGE,  False, pts, THICK_MOUTH)


class FaceDisplay:
    """
    Pygame FaceDisplay:
    - start() chạy nền
    - set_face() đổi mặt vĩnh viễn
    - push_face(ttl) đổi mặt tạm thời rồi quay lại
    """

    SUPPORTED = ["love_eyes", "music", "what_is_it", "suprise", "sleep", "sad", "angry"]

    def __init__(self, fps=60, default_face="music", fullscreen=True, hide_mouse=True):
        self.fps = max(10, int(fps))
        self.fullscreen = bool(fullscreen)
        self.hide_mouse = bool(hide_mouse)

        if default_face not in self.SUPPORTED:
            default_face = "music"

        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._screen = None
        self._clock = None

        self._face = default_face
        self._stack = []
        self._push_until = 0.0

        self._should_quit = False

    # ----- public API -----
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            pygame.quit()
        except Exception:
            pass

    def set_face(self, face: str):
        if face not in self.SUPPORTED:
            return
        with self._lock:
            self._face = face
            self._stack.clear()
            self._push_until = 0.0

    def push_face(self, face: str, ttl: float = 1.2):
        if face not in self.SUPPORTED:
            return
        ttl = max(0.05, float(ttl))
        now = time.monotonic()
        with self._lock:
            self._stack.append(self._face)
            self._face = face
            self._push_until = now + ttl

    def should_quit(self) -> bool:
        return self._should_quit

    # Optional: nếu bạn muốn chạy loop ngay trong main (không thread)
    def run_forever(self):
        self._init_pygame()
        self._run_loop()

    # ----- internal -----
    def _init_pygame(self):
        pygame.init()
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        self._screen = pygame.display.set_mode((0, 0), flags)
        pygame.display.set_caption("Robot FaceDisplay")
        if self.hide_mouse:
            pygame.mouse.set_visible(False)
        self._clock = pygame.time.Clock()

    def _poll_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self._should_quit = True
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._should_quit = True

    def _loop(self):
        # Thread entry
        self._init_pygame()
        self._run_loop()

    def _run_loop(self):
        while self._running and not self._should_quit:
            now = time.monotonic()

            # handle push timeout
            with self._lock:
                if self._push_until > 0.0 and now >= self._push_until:
                    if self._stack:
                        self._face = self._stack.pop()
                    self._push_until = 0.0
                face = self._face

            self._poll_events()
            render_face(self._screen, face, time.time())  # use wall time for smooth motion
            pygame.display.flip()
            self._clock.tick(self.fps)

        self._running = False
        try:
            pygame.quit()
        except Exception:
            pass


# --------------------------
# Example main usage
# --------------------------
if __name__ == "__main__":
    face = FaceDisplay(fps=60, default_face="music", fullscreen=True)
    face.start()

    try:
        # Demo: giả lập output class khác
        while not face.should_quit():
            face.set_face("music")
            time.sleep(2)

            face.push_face("suprise", ttl=1.0)  # event tạm
            time.sleep(2)

            face.set_face("love_eyes")
            time.sleep(2)

            face.set_face("sad")
            time.sleep(2)

            face.push_face("angry", ttl=1.3)
            time.sleep(2)

    except KeyboardInterrupt:
        pass
    finally:
        face.stop()
