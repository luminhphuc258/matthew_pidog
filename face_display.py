#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np


class FaceDisplay:
    """
    Vẽ mặt kiểu robot (happy/sad/angry/neutral) bằng OpenCV window.
    Nếu bạn chạy trên màn hình Pi (có X11/desktop) sẽ thấy.
    """

    def __init__(self, win_name="MatthewFace", w=480, h=320, fullscreen=False):
        self.win_name = win_name
        self.w, self.h = w, h
        self.fullscreen = fullscreen
        self.emotion = "neutral"
        self._last_draw = 0.0

        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.w, self.h)
        if self.fullscreen:
            cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def set_emotion(self, emo: str):
        self.emotion = emo

    def _draw(self) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # face background
        cv2.rectangle(img, (0, 0), (self.w, self.h), (20, 20, 20), -1)

        # eyes base positions
        cx1, cx2 = int(self.w * 0.35), int(self.w * 0.65)
        cy = int(self.h * 0.45)

        eye_w, eye_h = int(self.w * 0.12), int(self.h * 0.18)

        def glow_eye(cx, cy, tilt=0):
            pts = np.array([
                [cx - eye_w, cy - eye_h],
                [cx + eye_w, cy - eye_h],
                [cx + eye_w, cy + eye_h],
                [cx - eye_w, cy + eye_h],
            ], dtype=np.float32)
            # tilt
            M = cv2.getRotationMatrix2D((cx, cy), tilt, 1.0)
            pts2 = cv2.transform(np.array([pts]), M)[0].astype(np.int32)
            cv2.fillPoly(img, [pts2], (0, 140, 255))  # amber-ish
            cv2.polylines(img, [pts2], True, (0, 180, 255), 2)

        if self.emotion == "happy":
            glow_eye(cx1, cy, tilt=0)
            glow_eye(cx2, cy, tilt=0)
            cv2.ellipse(img, (self.w//2, int(self.h*0.72)), (80, 35), 0, 10, 170, (0, 180, 255), 4)

        elif self.emotion == "sad":
            glow_eye(cx1, cy+10, tilt=-10)
            glow_eye(cx2, cy+10, tilt=10)
            cv2.ellipse(img, (self.w//2, int(self.h*0.75)), (70, 30), 0, 200, 340, (0, 180, 255), 4)

        elif self.emotion == "angry":
            glow_eye(cx1, cy, tilt=18)
            glow_eye(cx2, cy, tilt=-18)
            cv2.line(img, (int(self.w*0.42), int(self.h*0.68)), (int(self.w*0.58), int(self.h*0.68)), (0, 180, 255), 5)

        else:
            glow_eye(cx1, cy, tilt=0)
            glow_eye(cx2, cy, tilt=0)

        return img

    def tick(self, fps=15):
        now = time.time()
        if now - self._last_draw < 1.0 / fps:
            cv2.waitKey(1)
            return

        self._last_draw = now
        img = self._draw()
        cv2.imshow(self.win_name, img)
        cv2.waitKey(1)

    def close(self):
        try:
            cv2.destroyWindow(self.win_name)
        except Exception:
            pass
