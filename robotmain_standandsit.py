#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import subprocess
import random
from pathlib import Path

from perception_planner import PerceptionPlanner
from motion_controller import MotionController
from face_display import FaceDisplay
from web_dashboard import WebDashboard
from active_listener import ActiveListener

import numpy as np

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

MIC_DEVICE = "default"
SPK_DEVICE = "default"


def run(cmd):
    return subprocess.run(cmd, check=False)


def set_volumes():
    run(["amixer", "-q", "sset", "robot-hat speaker", "100%"])
    run(["amixer", "-q", "sset", "robot-hat speaker Playback Volume", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic Capture Volume", "100%"])


def json_safe(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def main():
    # reduce audio backend conflicts
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # 1) perception (CAM + MQTT) -> chỉ để hiển thị dashboard / status
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,

        safe_dist_cm=50.0,
        emergency_stop_cm=10.0,

        enable_camera=True,
        enable_imu=False,

        enable_mqtt=True,
        mqtt_host="rfff7184.ala.us-east-1.emqxsl.com",
        mqtt_port=8883,
        mqtt_user="robot_matthew",
        mqtt_pass="29061992abCD!yesokmen",
        mqtt_topic="/pidog/sensorhubdata",
        mqtt_client_id="pidog-perception-pi",
        mqtt_debug=False,
        mqtt_insecure=True,
    )
    planner.start()

    # 2) motion
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()
    set_volumes()

    # ✅ luôn đứng yên sau boot
    try:
        motion.execute("STOP")
    except Exception:
        pass

    # 3) face
    face = FaceDisplay(default_face="what_is_it", fps=60, fullscreen=True)
    face.start()

    # 4) active listening (chỉ để nghe/đẩy lên server, không chặn motion)
    listener = ActiveListener(
        mic_device=MIC_DEVICE,
        speaker_device=SPK_DEVICE,
        threshold=2500,
        cooldown_sec=1.0,
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        debug=False,
    )
    listener.start()

    # ======================
    # MANUAL OVERRIDE (web)
    # ======================
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = (move or "STOP").upper()
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    # ======================
    # IDLE ANIMATION
    # ======================
    # thỉnh thoảng sit->stand, thỉnh thoảng push up
    next_idle_ts = time.time() + random.uniform(8.0, 16.0)

    def do_idle_action():
        """
        Ưu tiên an toàn: chỉ làm action khi KHÔNG có manual.
        Action nhẹ, không di chuyển tới.
        """
        nonlocal next_idle_ts
        r = random.random()

        # 70% sit->stand, 30% push_up
        try:
            if r < 0.70:
                face.set_face("what_is_it")
                # sit
                motion.execute("SIT")
                time.sleep(1.0)
                # stand
                motion.execute("STAND")
            else:
                face.set_face("music")
                motion.execute("PUSH_UP")
                time.sleep(0.3)
                motion.execute("STAND")
        except Exception:
            # nếu MotionController chưa support SIT/STAND/PUSH_UP
            # thì bỏ qua, vẫn không crash
            pass

        # schedule lần tiếp theo
        next_idle_ts = time.time() + random.uniform(10.0, 22.0)

    # status builder for web
    def build_status():
        st = planner.get_status_dict()
        st.update({
            "mode": "IDLE",
            "manual_move": manual["move"],
            "manual_active": bool(manual_active()),
            "listener_transcript": listener.last_transcript,
            "listener_label": listener.last_label,
            "mic_device": MIC_DEVICE,
            "spk_device": SPK_DEVICE,
        })
        return json_safe(st)

    # 5) web dashboard
    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_jpeg=lambda: planner.latest_jpeg,
        get_status=build_status,
        get_minimap_png=planner.get_mini_map_png,
        on_manual_cmd=on_manual_cmd,
    )
    threading.Thread(target=web.run, daemon=True).start()

    # ======================
    # MAIN LOOP
    # ======================
    try:
        while True:
            # nếu có manual -> robot chỉ làm manual
            if manual_active():
                decision = manual["move"]
                # face theo manual
                if decision == "FORWARD":
                    face.set_face("music")
                elif decision in ("TURN_LEFT", "TURN_RIGHT", "LEFT", "RIGHT"):
                    face.set_face("what_is_it")
                elif decision == "BACK":
                    face.set_face("angry")
                else:
                    face.set_face("what_is_it")

                motion.execute(decision)

            else:
                # không manual -> đứng yên + idle action theo lịch
                manual["move"] = None
                try:
                    motion.execute("STOP")
                except Exception:
                    pass

                # face idle
                face.set_face("what_is_it")

                # thỉnh thoảng sit/stand/pushup
                if time.time() >= next_idle_ts:
                    do_idle_action()

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")

    finally:
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
