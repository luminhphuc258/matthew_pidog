#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np


class FaceDisplay:
    """
    Nếu có DISPLAY => dùng cv2.imshow.
    Nếu không có DISPLAY (SSH/sudo) => headless: chỉ render ảnh ra file /tmp/face.png
    """
    def __init__(self, win_name="MatthewFace", w=480, h=320, fullscreen=False, headless=None, out_path="/tmp/face.png"):
        self.win_name = win_name
        self.w, self.h = w, h
        self.fullscreen = fullscreen
        self.emotion = "neutral"
        self._last_draw = 0.0
        self.out_path = out_path

        if headless is None:
            headless = (os.environ.get("DISPLAY") is None)
        self.headless = headless

        if not self.headless:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win_name, self.w, self.h)
            if self.fullscreen:
                cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def set_emotion(self, emo: str):
        self.emotion = emo

    def _draw(self) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (self.w, self.h), (20, 20, 20), -1)

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
            M = cv2.getRotationMatrix2D((cx, cy), tilt, 1.0)
            pts2 = cv2.transform(np.array([pts]), M)[0].astype(np.int32)
            cv2.fillPoly(img, [pts2], (0, 140, 255))
            cv2.polylines(img, [pts2], True, (0, 180, 255), 2)

        if self.emotion == "happy":
            glow_eye(cx1, cy, 0); glow_eye(cx2, cy, 0)
            cv2.ellipse(img, (self.w//2, int(self.h*0.72)), (80, 35), 0, 10, 170, (0, 180, 255), 4)
        elif self.emotion == "sad":
            glow_eye(cx1, cy+10, -10); glow_eye(cx2, cy+10, 10)
            cv2.ellipse(img, (self.w//2, int(self.h*0.75)), (70, 30), 0, 200, 340, (0, 180, 255), 4)
        elif self.emotion == "angry":
            glow_eye(cx1, cy, 18); glow_eye(cx2, cy, -18)
            cv2.line(img, (int(self.w*0.42), int(self.h*0.68)), (int(self.w*0.58), int(self.h*0.68)), (0, 180, 255), 5)
        else:
            glow_eye(cx1, cy, 0); glow_eye(cx2, cy, 0)

        return img

    def tick(self, fps=15):
        now = time.time()
        if now - self._last_draw < 1.0 / fps:
            if not self.headless:
                cv2.waitKey(1)
            return

        self._last_draw = now
        img = self._draw()

        if self.headless:
            # lưu ra file để debug / hoặc bạn dùng web đọc file này
            try:
                cv2.imwrite(self.out_path, img)
            except Exception:
                pass
            return

        cv2.imshow(self.win_name, img)
        cv2.waitKey(1)

    def close(self):
        if not self.headless:
            try:
                cv2.destroyWindow(self.win_name)
            except Exception:
                pass
