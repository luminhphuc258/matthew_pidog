#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import socket
from pathlib import Path

import cv2

from handcommand import HandCommand, HandCfg
from web_dashboard import WebDashboard

# ===== face UDP (giống cách bạn set face3d) =====
FACE_HOST = "127.0.0.1"
FACE_PORT = 39393
_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def set_face(emo: str):
    try:
        _face_sock.sendto(emo.encode("utf-8"), (FACE_HOST, FACE_PORT))
    except Exception:
        pass


# ===== camera (shared) =====
class Camera:
    def __init__(self, dev="/dev/video0", w=640, h=480, fps=30):
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.last = None
        self.ts = 0.0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last = frame
        self.ts = time.time()
        return frame

    def get_frame(self):
        # WebDashboard gọi hàm này liên tục
        # -> nếu chưa có frame mới thì đọc thêm
        fr = self.read()
        return fr if fr is not None else self.last

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


def main():
    cam = Camera(dev="/dev/video0", w=640, h=480, fps=30)

    # ===== callback khi HandCommand detect gesture =====
    def on_action(action: str, face: str, bark: bool):
        print(f"[CMD ] {action}")
        print(f"[FACE] {face}  bark={bark}")
        set_face(face)

        # NOTE:
        # ở test này mình chỉ print + set face.
        # sau này bạn thay bằng MotionController: move(action)
        # bark=True thì bạn phát âm thanh sủa

    # ===== HandCommand =====
    hc = HandCommand(
        cfg=HandCfg(
            cam_dev="/dev/video0",
            w=640, h=480, fps=30,
            process_every=2,          # ✅ nhẹ hơn, đỡ lag
            action_cooldown_sec=0.7
        ),
        on_action=on_action,
        boot_helper=None,            # nếu muốn support_stand thật: truyền MatthewPidogBootClass()
        get_frame_bgr=cam.get_frame, # ✅ dùng chung camera với web
        open_own_camera=False,       # ✅ không tự mở camera nữa
        clear_memory_on_start=True
    )

    hc.start()
    hc.set_enabled(True)  # start ON luôn (bạn có thể OFF nếu muốn)

    # ===== WebDashboard =====
    dash = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_frame_bgr=cam.get_frame,
        avoid_obstacle=None,
        on_manual_cmd=lambda m: print("[MANUAL CMD]", m),
        rotate180=True,
        mqtt_enable=False,
        hand_command=hc,   # ✅ NEW
    )

    print("\n=== HandCommand + WebDashboard ===")
    print("Open browser: http://<pi_ip>:8000")
    print("Toggle Hand Command to see landmarks/gesture overlay.\n")

    try:
        dash.run()
    finally:
        hc.stop()
        cam.close()


if __name__ == "__main__":
    main()
