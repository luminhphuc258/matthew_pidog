#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np

class FaceDisplay:
    """
    Vẽ ảnh RGB trực tiếp lên framebuffer (/dev/fb0).
    Không cần X11, không dính Qt/xcb.
    """

    def __init__(self, fb_path="/dev/fb0", width=None, height=None):
        self.fb_path = fb_path

        # đọc size framebuffer từ sysfs nếu user không set
        if width is None or height is None:
            w, h = self._read_fb_size()
            width = width or w
            height = height or h

        self.w = int(width)
        self.h = int(height)

        # detect bpp
        self.bpp = self._read_bpp()  # usually 16 or 32
        if self.bpp not in (16, 32):
            raise RuntimeError(f"Unsupported framebuffer bpp={self.bpp}. Only 16/32 supported.")

        self.emotion = "neutral"

    def _read_fb_size(self):
        # /sys/class/graphics/fb0/virtual_size -> "800,480"
        try:
            s = open("/sys/class/graphics/fb0/virtual_size", "r").read().strip()
            w, h = s.split(",")
            return int(w), int(h)
        except Exception:
            # fallback phổ biến waveshare
            return 800, 480

    def _read_bpp(self):
        # /sys/class/graphics/fb0/bits_per_pixel
        try:
            return int(open("/sys/class/graphics/fb0/bits_per_pixel", "r").read().strip())
        except Exception:
            return 16

    def set_emotion(self, emo: str):
        self.emotion = emo

    def _render_face(self):
        """
        Render mặt robot (happy/sad/angry/neutral) ra numpy RGB (H,W,3)
        """
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (20, 20, 20)

        # eyes
        cx1, cx2 = int(self.w * 0.35), int(self.w * 0.65)
        cy = int(self.h * 0.45)
        eye_w, eye_h = int(self.w * 0.08), int(self.h * 0.14)

        def draw_rect(img, x0, y0, x1, y1, color):
            x0 = max(0, min(self.w-1, x0)); x1 = max(0, min(self.w, x1))
            y0 = max(0, min(self.h-1, y0)); y1 = max(0, min(self.h, y1))
            img[y0:y1, x0:x1] = color

        amber = (255, 170, 0)  # RGB
        def eye(cx, tilt=0):
            # tilt đơn giản: dịch lên/xuống góc thay vì rotate thật
            y_shift = int(tilt / 2)
            draw_rect(img, cx-eye_w, cy-eye_h+y_shift, cx+eye_w, cy+eye_h+y_shift, amber)

        if self.emotion == "happy":
            eye(cx1, 0); eye(cx2, 0)
            # mouth smile
            mx0, mx1 = int(self.w*0.40), int(self.w*0.60)
            my = int(self.h*0.72)
            draw_rect(img, mx0, my, mx1, my+10, amber)

        elif self.emotion == "sad":
            eye(cx1, -10); eye(cx2, 10)
            mx0, mx1 = int(self.w*0.42), int(self.w*0.58)
            my = int(self.h*0.75)
            draw_rect(img, mx0, my, mx1, my+10, amber)

        elif self.emotion == "angry":
            eye(cx1, 18); eye(cx2, -18)
            mx0, mx1 = int(self.w*0.45), int(self.w*0.55)
            my = int(self.h*0.68)
            draw_rect(img, mx0, my, mx1, my+14, amber)

        else:
            eye(cx1, 0); eye(cx2, 0)

        return img

    def _rgb_to_rgb565(self, rgb):
        r = (rgb[..., 0] >> 3).astype(np.uint16)
        g = (rgb[..., 1] >> 2).astype(np.uint16)
        b = (rgb[..., 2] >> 3).astype(np.uint16)
        return (r << 11) | (g << 5) | b

    def tick(self):
        img = self._render_face()  # RGB

        with open(self.fb_path, "wb") as f:
            if self.bpp == 32:
                # convert to BGRA (framebuffer hay dùng BGRA)
                bgr = img[..., ::-1]
                alpha = np.full((self.h, self.w, 1), 255, dtype=np.uint8)
                bgra = np.concatenate([bgr, alpha], axis=2)
                f.write(bgra.tobytes())
            else:
                # 16bpp RGB565
                rgb565 = self._rgb_to_rgb565(img)
                f.write(rgb565.tobytes())
