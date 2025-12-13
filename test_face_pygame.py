#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time
import pygame

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# =========================
# 9 emotions (full-screen)
# =========================
# 1: Joy (tongue out)
# 2: Angry (arms crossed)
# 3: Big Laugh (open mouth)
# 4: Surprise (hands up)
# 5: Shy (hands clasp)
# 6: Love (big smile)
# 7: Bleh (tongue long)
# 8: Cry (hands on cheeks)
# 9: Shock (hands near cheeks, mouth small)
EMO_KEYS = ["joy","angry","laugh","surprise","shy","love","bleh","cry","shock"]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_round_rect(surf, color, rect, radius=12, width=0):
    pygame.draw.rect(surf, color, rect, width, border_radius=radius)


class DoraFaceRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.W, self.H = screen.get_size()

        # palette
        self.BG = (200, 230, 250)          # light blue
        self.BLUE = (60, 165, 225)
        self.WHITE = (245, 245, 245)
        self.BLACK = (25, 25, 25)
        self.RED = (235, 65, 65)
        self.PINK = (255, 170, 190)
        self.MOUTH_RED = (200, 50, 55)
        self.TONGUE = (255, 135, 125)
        self.COLLAR = (230, 40, 45)
        self.BELL = (245, 210, 75)

        self.cx = self.W // 2
        self.cy = int(self.H * 0.47)
        self.size = int(min(self.W, self.H) * 0.78)

        self.head_r = int(self.size * 0.34)
        self.face_r = int(self.size * 0.27)

        # eye geometry
        self.eye_w = int(self.size * 0.11)
        self.eye_h = int(self.size * 0.15)
        self.eye_gap = int(self.size * 0.03)
        self.eye_y = self.cy - int(self.size * 0.15)

        self.nose_r = int(self.size * 0.040)
        self.nose_c = (self.cx, self.cy - int(self.size * 0.02))

        # animation
        self.t0 = time.time()

    def pupil_offset(self, mode: str, t: float):
        """
        Animated eye movement. Returns (dx, dy) in pixels for pupil.
        Different modes have different "personality".
        """
        base = int(self.size * 0.012)
        # global subtle drift left-right
        dx = int(math.sin(t * 1.8) * base)
        dy = int(math.sin(t * 1.1 + 1.7) * (base // 2))

        if mode in ("angry",):
            dx += int(math.sin(t * 6.0) * (base // 2))
            dy += int(math.sin(t * 7.0) * (base // 3))
        elif mode in ("cry", "shock"):
            dy += int(abs(math.sin(t * 3.5)) * base)
        elif mode in ("shy",):
            dx -= int(abs(math.sin(t * 2.2)) * base)
            dy += int(abs(math.sin(t * 2.6)) * (base // 2))
        elif mode in ("love", "laugh"):
            dx += int(math.sin(t * 2.6) * base)
        elif mode in ("bleh", "joy"):
            dx += int(math.sin(t * 3.0) * base)
            dy -= int(math.sin(t * 2.4) * (base // 2))

        return dx, dy

    def _draw_head_base(self):
        s = self.screen
        s.fill(self.BG)

        # head blue
        pygame.draw.circle(s, self.BLUE, (self.cx, self.cy), self.head_r)

        # face white
        pygame.draw.circle(s, self.WHITE, (self.cx, self.cy + int(self.size*0.02)), self.face_r)

        # nose
        pygame.draw.circle(s, self.RED, self.nose_c, self.nose_r)
        pygame.draw.circle(s, (255,255,255), (self.nose_c[0]-int(self.nose_r*0.35), self.nose_c[1]-int(self.nose_r*0.35)),
                           max(2, int(self.nose_r*0.25)))

        # center line
        pygame.draw.line(s, self.BLACK,
                         (self.cx, self.cy + int(self.size*0.01)),
                         (self.cx, self.cy + int(self.size*0.22)), 3)

        # cheeks blush
        blush_r = int(self.size*0.045)
        pygame.draw.circle(s, self.PINK, (self.cx - int(self.size*0.18), self.cy + int(self.size*0.05)), blush_r)
        pygame.draw.circle(s, self.PINK, (self.cx + int(self.size*0.18), self.cy + int(self.size*0.05)), blush_r)

        # whiskers
        whisk_y0 = self.cy + int(self.size*0.02)
        for i in (-1, 0, 1):
            yy = whisk_y0 + i * int(self.size*0.06)
            pygame.draw.line(s, self.BLACK,
                             (self.cx - int(self.size*0.28), yy),
                             (self.cx - int(self.size*0.07), yy), 3)
            pygame.draw.line(s, self.BLACK,
                             (self.cx + int(self.size*0.07), yy),
                             (self.cx + int(self.size*0.28), yy), 3)

        # collar + bell
        collar_y = self.cy + int(self.size*0.33)
        collar_h = int(self.size*0.055)
        collar_rect = pygame.Rect(self.cx - int(self.size*0.26), collar_y, int(self.size*0.52), collar_h)
        draw_round_rect(s, self.COLLAR, collar_rect, radius=int(collar_h*0.45))

        bell_c = (self.cx, collar_y + collar_h + int(self.size*0.03))
        bell_r = int(self.size*0.05)
        pygame.draw.circle(s, self.BELL, bell_c, bell_r)
        pygame.draw.circle(s, self.BLACK, bell_c, bell_r, 2)
        pygame.draw.line(s, self.BLACK, (bell_c[0]-bell_r, bell_c[1]-int(bell_r*0.2)),
                         (bell_c[0]+bell_r, bell_c[1]-int(bell_r*0.2)), 2)
        pygame.draw.circle(s, self.BLACK, (bell_c[0], bell_c[1]+int(bell_r*0.25)), max(2, int(bell_r*0.12)))

    def _eye_rects(self):
        left = pygame.Rect(self.cx - self.eye_gap - self.eye_w, self.eye_y, self.eye_w, self.eye_h)
        right = pygame.Rect(self.cx + self.eye_gap, self.eye_y, self.eye_w, self.eye_h)
        return left, right

    def _draw_eyes_open(self, mode: str, t: float, squint=0):
        s = self.screen
        left, right = self._eye_rects()

        # squint shrinks height
        if squint != 0:
            shrink = int(abs(squint) * self.eye_h)
            left = pygame.Rect(left.x, left.y + shrink//2, left.w, max(8, left.h - shrink))
            right = pygame.Rect(right.x, right.y + shrink//2, right.w, max(8, right.h - shrink))

        pygame.draw.ellipse(s, self.WHITE, left)
        pygame.draw.ellipse(s, self.WHITE, right)
        pygame.draw.ellipse(s, self.BLACK, left, 3)
        pygame.draw.ellipse(s, self.BLACK, right, 3)

        dx, dy = self.pupil_offset(mode, t)
        pr = max(4, int(self.size*0.018))

        # pupils inside bounds
        lx = clamp(left.centerx + dx, left.left + pr + 3, left.right - pr - 3)
        ly = clamp(left.centery + dy, left.top + pr + 3, left.bottom - pr - 3)
        rx = clamp(right.centerx + dx, right.left + pr + 3, right.right - pr - 3)
        ry = clamp(right.centery + dy, right.top + pr + 3, right.bottom - pr - 3)

        pygame.draw.circle(s, self.BLACK, (lx, ly), pr)
        pygame.draw.circle(s, self.BLACK, (rx, ry), pr)

        # highlights
        pygame.draw.circle(s, (255,255,255), (lx-int(pr*0.4), ly-int(pr*0.4)), max(2, pr//3))
        pygame.draw.circle(s, (255,255,255), (rx-int(pr*0.4), ry-int(pr*0.4)), max(2, pr//3))

    def _draw_eyes_happy_closed(self):
        s = self.screen
        left, right = self._eye_rects()
        # arcs
        rect_l = pygame.Rect(left.x-2, left.y+int(left.h*0.35), left.w+4, int(left.h*0.7))
        rect_r = pygame.Rect(right.x-2, right.y+int(right.h*0.35), right.w+4, int(right.h*0.7))
        pygame.draw.arc(s, self.BLACK, rect_l, math.pi, 2*math.pi, 4)
        pygame.draw.arc(s, self.BLACK, rect_r, math.pi, 2*math.pi, 4)

    def _draw_brows_angry(self):
        s = self.screen
        left, right = self._eye_rects()
        pygame.draw.line(s, self.BLACK,
                         (left.left, left.top + int(left.h*0.25)),
                         (left.right, left.top - int(left.h*0.10)), 4)
        pygame.draw.line(s, self.BLACK,
                         (right.left, right.top - int(right.h*0.10)),
                         (right.right, right.top + int(right.h*0.25)), 4)

    def _mouth_open(self, big=True):
        s = self.screen
        w = int(self.size * (0.36 if big else 0.24))
        h = int(self.size * (0.22 if big else 0.16))
        rect = pygame.Rect(self.cx - w//2, self.cy + int(self.size*0.10), w, h)
        pygame.draw.ellipse(s, self.MOUTH_RED, rect)
        pygame.draw.ellipse(s, self.BLACK, rect, 4)

        # tongue
        tr = pygame.Rect(rect.x + int(rect.w*0.22), rect.y + int(rect.h*0.50), int(rect.w*0.56), int(rect.h*0.50))
        pygame.draw.ellipse(s, self.TONGUE, tr)
        pygame.draw.ellipse(s, self.BLACK, tr, 2)

    def _mouth_big_laugh(self):
        self._mouth_open(big=True)

    def _mouth_small_o(self):
        s = self.screen
        rect = pygame.Rect(self.cx - int(self.size*0.08), self.cy + int(self.size*0.12), int(self.size*0.16), int(self.size*0.18))
        pygame.draw.ellipse(s, self.MOUTH_RED, rect)
        pygame.draw.ellipse(s, self.BLACK, rect, 4)

    def _mouth_smile(self):
        s = self.screen
        rect = pygame.Rect(self.cx - int(self.size*0.18), self.cy + int(self.size*0.12), int(self.size*0.36), int(self.size*0.22))
        pygame.draw.arc(s, self.BLACK, rect, 0, math.pi, 5)

    def _draw_hands(self, pose: str):
        """
        Simple hands/arms in front.
        """
        s = self.screen
        hand_r = int(self.size*0.06)
        y = self.cy + int(self.size*0.22)

        if pose == "hands_up":  # surprise
            x1 = self.cx - int(self.size*0.22)
            x2 = self.cx + int(self.size*0.22)
            pygame.draw.circle(s, self.WHITE, (x1, y), hand_r)
            pygame.draw.circle(s, self.WHITE, (x2, y), hand_r)
            pygame.draw.circle(s, self.BLACK, (x1, y), hand_r, 2)
            pygame.draw.circle(s, self.BLACK, (x2, y), hand_r, 2)

        elif pose == "clasp":  # shy
            x1 = self.cx - int(self.size*0.05)
            x2 = self.cx + int(self.size*0.05)
            pygame.draw.circle(s, self.WHITE, (x1, y), hand_r)
            pygame.draw.circle(s, self.WHITE, (x2, y), hand_r)
            pygame.draw.circle(s, self.BLACK, (x1, y), hand_r, 2)
            pygame.draw.circle(s, self.BLACK, (x2, y), hand_r, 2)

        elif pose == "cheeks":  # cry / shock
            y2 = self.cy + int(self.size*0.10)
            x1 = self.cx - int(self.size*0.24)
            x2 = self.cx + int(self.size*0.24)
            pygame.draw.circle(s, self.WHITE, (x1, y2), hand_r)
            pygame.draw.circle(s, self.WHITE, (x2, y2), hand_r)
            pygame.draw.circle(s, self.BLACK, (x1, y2), hand_r, 2)
            pygame.draw.circle(s, self.BLACK, (x2, y2), hand_r, 2)

        elif pose == "cross":  # angry arms crossed
            # two blue arms crossing
            arm_w = int(self.size*0.24)
            arm_h = int(self.size*0.08)
            r = int(arm_h*0.45)
            rect1 = pygame.Rect(self.cx - arm_w, self.cy + int(self.size*0.24), arm_w, arm_h)
            rect2 = pygame.Rect(self.cx, self.cy + int(self.size*0.24), arm_w, arm_h)
            draw_round_rect(s, self.BLUE, rect1, radius=r)
            draw_round_rect(s, self.BLUE, rect2, radius=r)

            # hands tips
            pygame.draw.circle(s, self.WHITE, (rect1.left+int(arm_h*0.5), rect1.centery), hand_r)
            pygame.draw.circle(s, self.WHITE, (rect2.right-int(arm_h*0.5), rect2.centery), hand_r)
            pygame.draw.circle(s, self.BLACK, (rect1.left+int(arm_h*0.5), rect1.centery), hand_r, 2)
            pygame.draw.circle(s, self.BLACK, (rect2.right-int(arm_h*0.5), rect2.centery), hand_r, 2)

    def _draw_tears(self, t: float):
        s = self.screen
        left, right = self._eye_rects()
        # tears under eyes
        for ex in (left.centerx, right.centerx):
            drop_w = int(self.size*0.035)
            drop_h = int(self.size*0.07)
            yy = left.bottom + int(self.size*0.02) + int(abs(math.sin(t*3.0))*6)
            rect = pygame.Rect(ex - drop_w//2, yy, drop_w, drop_h)
            pygame.draw.ellipse(s, (170, 220, 255), rect)
            pygame.draw.ellipse(s, (110, 170, 235), rect, 2)

    def render(self, emo: str):
        t = time.time() - self.t0

        self._draw_head_base()

        # 9 unique expressions (Dora-like)
        if emo == "joy":
            # squint + big mouth + tongue out
            self._draw_eyes_open(emo, t, squint=0.15)
            self._mouth_open(big=True)

        elif emo == "angry":
            self._draw_eyes_open(emo, t, squint=0.25)
            self._draw_brows_angry()
            self._mouth_smile()
            self._draw_hands("cross")

        elif emo == "laugh":
            self._draw_eyes_open(emo, t, squint=0.0)
            self._mouth_big_laugh()

        elif emo == "surprise":
            self._draw_eyes_open(emo, t, squint=0.0)
            self._mouth_small_o()
            self._draw_hands("hands_up")

        elif emo == "shy":
            self._draw_eyes_open(emo, t, squint=0.10)
            self._mouth_smile()
            self._draw_hands("clasp")

        elif emo == "love":
            # happy closed eyes + big smile mouth
            self._draw_eyes_happy_closed()
            self._mouth_big_laugh()
            self._draw_hands("clasp")

        elif emo == "bleh":
            self._draw_eyes_open(emo, t, squint=0.0)
            # long tongue
            self._mouth_open(big=False)
            s = self.screen
            tongue_rect = pygame.Rect(self.cx - int(self.size*0.06), self.cy + int(self.size*0.22), int(self.size*0.12), int(self.size*0.18))
            pygame.draw.ellipse(s, self.TONGUE, tongue_rect)
            pygame.draw.ellipse(s, self.BLACK, tongue_rect, 2)

        elif emo == "cry":
            self._draw_eyes_open(emo, t, squint=0.15)
            self._mouth_open(big=True)
            self._draw_hands("cheeks")
            self._draw_tears(t)

        elif emo == "shock":
            self._draw_eyes_open(emo, t, squint=0.0)
            self._mouth_small_o()
            self._draw_hands("cheeks")

        pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)

    renderer = DoraFaceRenderer(screen)

    idx = 0
    auto = True
    last_switch = time.time()

    print("Dora-like 9 emotions fullscreen")
    print("Keys: 1..9 switch | Space toggle auto | ESC quit")

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if e.key == pygame.K_SPACE:
                    auto = not auto
                # number keys 1..9
                if pygame.K_1 <= e.key <= pygame.K_9:
                    idx = e.key - pygame.K_1
                    auto = False

        if auto and (time.time() - last_switch) > 2.0:
            idx = (idx + 1) % 9
            last_switch = time.time()

        emo = EMO_KEYS[idx]
        renderer.render(emo)
        time.sleep(0.02)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
