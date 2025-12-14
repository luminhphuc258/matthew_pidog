#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
from pathlib import Path

from perception_planner import PerceptionPlanner
from motion_controller import MotionController
from face_display import FaceDisplay
from web_dashboard import WebDashboard
from active_listener import ActiveListener

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"


def main():
    print("[BOOT] Matthew PiDog – STATIONARY TEST MODE")

    # 1) perception (camera / lidar vẫn chạy)
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        serial_port="/dev/ttyUSB0",
        baud=115200,
        safe_dist_cm=50.0,
        enable_imu=False,
    )
    planner.start()

    # 2) motion (chỉ boot pose, KHÔNG di chuyển)
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()   # chỉ dựng dáng + khóa đầu

    # 3) face display (pygame thread)
    face = FaceDisplay(
        default_face="what_is_it",
        fps=60,
        fullscreen=True
    )
    face.start()

    # 4) active listening (mic + speaker)
    listener = ActiveListener(
    mic_device="default",
    speaker_device="default",
    threshold=2500,
    cooldown_sec=1.0,
    nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
    debug=False,
    )
    listener.start()

    # manual override from web (vẫn nhận nhưng CHƯA dùng)
    manual = {"move": None, "ts": 0.0}

    def on_manual_cmd(move: str):
        manual["move"] = move
        manual["ts"] = time.time()

    # 5) web dashboard
    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_jpeg=lambda: planner.latest_jpeg,
        get_status=lambda: {
            **planner.get_status_dict(),
            "manual_move": manual["move"],
            "listener_transcript": listener.last_transcript,
            "listener_label": listener.last_label,
            "mode": "STATIONARY_TEST",
        },
        get_minimap_png=planner.get_mini_map_png,
        on_manual_cmd=on_manual_cmd,
    )
    threading.Thread(target=web.run, daemon=True).start()

    # =========================
    # MAIN LOOP – NO MOVEMENT
    # =========================
    try:
        while True:
            face.set_face("what_is_it")
            st = planner.get_state()

            # ❌ KHÔNG di chuyển
            # decision luôn coi như STOP
            decision = "STOP"

            # face logic (chỉ để test hiển thị)
            if st.cam_blocked:
                face.set_face("angry")
            elif listener.last_label in ("music", "singing"):
                face.set_face("music")
            elif listener.last_label in ("happy", "laugh"):
                face.set_face("love_eyes")
            elif listener.last_label in ("sleep",):
                face.set_face("sleep")
            else:
                face.set_face("what_is_it")

            # ❌ KHÔNG gọi motion.execute()
            # motion.execute(decision)  <-- cố tình bỏ

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    finally:
        print("[SHUTDOWN] Cleaning up...")
        listener.stop()
        listener.join(2.0)
        planner.stop()
        motion.close()
        face.stop()


if __name__ == "__main__":
    main()
