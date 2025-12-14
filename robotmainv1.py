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
    # 1) perception
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        serial_port="/dev/ttyUSB0",
        baud=115200,
        safe_dist_cm=50.0,
        emergency_stop_cm=10.0,   # nếu bạn có param này
        enable_imu=False,
        enable_camera=True,
        uart_debug=False,
        uart_filter_window=5,
    )
    planner.start()

    # 2) motion
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    # 3) face
    face = FaceDisplay(default_face="what_is_it", fps=60, fullscreen=True)
    face.start()

    # 4) active listening
    listener = ActiveListener(
        mic_device="plughw:4,0",
        speaker_device="plughw:3,0",
        threshold=2500,
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
    )
    listener.start()

    # manual override
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = move
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    # 5) web dashboard
    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_jpeg=lambda: planner.latest_jpeg,
        get_status=lambda: {
            **planner.get_status_dict(),
            "manual_move": manual["move"],
            "manual_active": bool(manual_active()),
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

            # choose decision
            decision = st.decision
            if manual_active():
                decision = manual["move"]
            else:
                manual["move"] = None

            # safety override (nếu distance quá gần)
            # dùng uart_dist_cm nếu có, fallback uart_dist_raw_cm
            dist = st.uart_dist_cm if getattr(st, "uart_dist_cm", None) is not None else getattr(st, "uart_dist_raw_cm", None)
            if dist is not None and dist < 10.0:
                decision = "STOP"

            # face mapping
            if decision == "FORWARD":
                face.set_face("music")
            elif decision in ("TURN_LEFT", "TURN_RIGHT"):
                face.set_face("what_is_it")
            elif decision in ("BACK",):
                face.set_face("angry")
            else:
                face.set_face("sad" if (st.cam_blocked or st.imu_bump) else "what_is_it")

            # ACT (giống main của bạn: gọi liên tục)
            motion.execute(decision)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        try:
            planner.stop()
        except Exception:
            pass
        try:
            motion.close()
        except Exception:
            pass
        try:
            face.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
