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
    print("[BOOT] Matthew PiDog – AUTO NAV MODE (Camera + UART Distance)")

    # 1) perception
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        serial_port="/dev/ttyUSB0",
        baud=115200,
        safe_dist_cm=50.0,          # tránh vật cản khi < 50cm
        emergency_stop_cm=10.0,     # STOP khẩn cấp khi < 10cm
        enable_imu=False,
        enable_camera=True,
        uart_debug=False,
        uart_filter_window=5,
    )
    planner.start()

    # 2) motion
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    # 3) face display
    face = FaceDisplay(
        default_face="what_is_it",
        fps=60,
        fullscreen=True
    )
    face.start()

    # 4) active listening
    listener = ActiveListener(
        mic_device="default",
        speaker_device="default",
        threshold=2500,
        cooldown_sec=1.0,
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        debug=False,
    )
    listener.start()

    # manual override window
    manual = {"move": None, "ts": 0.0}
    MANUAL_HOLD_SEC = 1.0

    def on_manual_cmd(move: str):
        manual["move"] = move
        manual["ts"] = time.time()

    def manual_active() -> bool:
        return manual["move"] is not None and (time.time() - manual["ts"]) <= MANUAL_HOLD_SEC

    # 5) web dashboard
    def status_payload():
        st = planner.get_status_dict()
        st.update({
            "manual_move": manual["move"],
            "manual_active": manual_active(),
            "listener_transcript": listener.last_transcript,
            "listener_label": listener.last_label,
            "mode": "AUTO_NAV",
        })
        return st

    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_jpeg=lambda: planner.latest_jpeg,
        get_status=status_payload,
        get_minimap_png=planner.get_mini_map_png,
        on_manual_cmd=on_manual_cmd,
    )
    threading.Thread(target=web.run, daemon=True).start()

    # =========================
    # MAIN LOOP – AUTO MOVE
    # =========================
    last_decision = "STOP"

    try:
        while True:
            st = planner.get_state()

            # decision from planner (combined camera + uart)
            auto_decision = st.decision

            # manual override (nhưng vẫn bị EMERGENCY_STOP override để an toàn)
            decision = auto_decision
            if manual_active():
                decision = manual["move"] or auto_decision

            # safety: if emergency stop condition happens, force STOP
            if st.uart_dist_cm is not None and st.uart_dist_cm < planner.emergency_stop_cm:
                decision = "STOP"

            # face logic
            if decision == "STOP" and st.uart_dist_cm is not None and st.uart_dist_cm < planner.safe_dist_cm:
                face.set_face("angry")
            elif listener.last_label in ("music", "singing"):
                face.set_face("music")
            elif listener.last_label in ("happy", "laugh"):
                face.set_face("love_eyes")
            elif listener.last_label in ("sleep",):
                face.set_face("sleep")
            else:
                face.set_face("what_is_it")

            # execute motion when decision changes or periodic
            if decision != last_decision:
                motion.execute(decision)
                last_decision = decision

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
