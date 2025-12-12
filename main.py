#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
from pathlib import Path

from perception_planner import PerceptionPlanner
from motion_controller import MotionController
from face_display import FaceDisplay
from web_dashboard import WebDashboard
from robot.active_listener import ActiveListener


POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"


def main():
    # 1) perception
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        serial_port="/dev/ttyUSB0",
        baud=115200,
        safe_dist_cm=50.0,
        enable_imu=False,  # bật True khi bạn đã đọc IMU thật
    )
    planner.start()

    # 2) motion (apply pose -> boot -> head lock)
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    # 3) face display
    face = FaceDisplay(w=480, h=320, fullscreen=False)

    # 5) active listening
    listener = ActiveListener(
        mic_device="plughw:4,0",
        speaker_device="plughw:3,0",
        threshold=2500,
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
    )
    listener.start()

    # manual override from web
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = move
        manual["ts"] = time.time()

    # 4) web dashboard
    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_jpeg=lambda: planner.latest_jpeg,
        get_status=lambda: {
            **planner.get_status_dict(),
            "manual_move": manual["move"],
            "listener_transcript": listener.last_transcript,
            "listener_label": listener.last_label,
        },
        get_minimap_png=planner.get_mini_map_png,
        on_manual_cmd=on_manual_cmd,
    )
    threading.Thread(target=web.run, daemon=True).start()

    # main loop
    try:
        while True:
            st = planner.get_state()

            # chọn decision: manual nếu còn hiệu lực
            decision = st.decision
            if manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec):
                decision = manual["move"]
            else:
                manual["move"] = None

            # emotion mapping
            if decision == "FORWARD":
                face.set_emotion("happy")
            elif decision in ("TURN_LEFT", "TURN_RIGHT"):
                face.set_emotion("neutral")
            elif decision in ("BACK",):
                face.set_emotion("angry")
            else:
                face.set_emotion("sad" if st.cam_blocked or st.imu_bump else "neutral")

            # act
            motion.execute(decision)

            # tick face UI
            face.tick(fps=15)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    finally:
        listener.stop()
        planner.stop()
        motion.close()
        face.close()


if __name__ == "__main__":
    main()
