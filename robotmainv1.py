#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import subprocess
from pathlib import Path

from perception_planner import PerceptionPlanner
from motion_controller import MotionController
from face_display import FaceDisplay
from web_dashboard import WebDashboard
from active_listener import ActiveListener

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# ✅ dùng default theo /etc/asound.conf của bạn
MIC_DEVICE = "default"
SPK_DEVICE = "default"


def run(cmd):
    return subprocess.run(cmd, check=False)


def set_volumes():
    """
    Cố gắng set volume cho robot-hat speaker/mic nếu control tồn tại.
    Không crash nếu không có.
    """
    run(["amixer", "-q", "sset", "robot-hat speaker", "100%"])
    run(["amixer", "-q", "sset", "robot-hat speaker Playback Volume", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic Capture Volume", "100%"])


def main():
    # (optional) giảm conflict audio backend
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")  # pygame
    os.environ.setdefault("PULSE_SERVER", "")         # đỡ kéo pulseaudio remote
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # 1) perception
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        safe_dist_cm=50.0,
        emergency_stop_cm=10.0,
        enable_imu=False,
        enable_camera=True,

        # ✅ MQTT config (đúng như bạn đang dùng trên N8R8)
        enable_mqtt=True,
        mqtt_host="rfff7184.ala.us-east-1.emqxsl.com",
        mqtt_port=8883,
        mqtt_user="robot_matthew",
        mqtt_pass="29061992abCD!yesokmen",
        mqtt_topic="/pidog/sensorhubdata",
        mqtt_client_id="pidog-perception-pi",
        mqtt_debug=False,
        )
    planner.start()

    # 2) motion
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()

    # ✅ sau boot: set volume giống script test (quan trọng)
    set_volumes()

    # 3) face
    face = FaceDisplay(default_face="what_is_it", fps=60, fullscreen=True)
    face.start()

    # 4) active listening (✅ dùng default giống script record/play OK)
    listener = ActiveListener(
        mic_device=MIC_DEVICE,
        speaker_device=SPK_DEVICE,
        threshold=2500,
        cooldown_sec=1.0,  # nếu class bạn có param này -> giúp ko thu tiếng robot
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        debug=False,
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
            "mic_device": MIC_DEVICE,
            "spk_device": SPK_DEVICE,
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

            # ACT
            motion.execute(decision)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")

    finally:
        # shutdown order: stop listener first (release audio devices)
        try:
            listener.stop()
            listener.join(2.0)
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
