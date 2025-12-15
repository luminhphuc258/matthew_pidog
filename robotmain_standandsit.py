#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import subprocess
import socket
import random
from pathlib import Path

import numpy as np

from perception_planner import PerceptionPlanner
from motion_controller import MotionController
from web_dashboard import WebDashboard
from active_listener import ActiveListener

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

MIC_DEVICE = "default"
SPK_DEVICE = "default"

# ===== FLOW TIMING =====
WAIT_IDLE_SEC = 9.0
ACTION_SEC = 2.0   # thời gian thực thi 1 hành động random

# ===== MANUAL OVERRIDE =====
manual = {"move": None, "ts": 0.0}
manual_timeout_sec = 1.2


# ========= FACE SERVICE (UDP) =========
FACE_HOST = "127.0.0.1"
FACE_PORT = 39393
_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def set_face(emo: str):
    """Gửi lệnh đổi mặt cho face3d_service.py (đang chạy bằng systemd)."""
    try:
        _face_sock.sendto(emo.encode("utf-8"), (FACE_HOST, FACE_PORT))
    except Exception:
        pass


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

    # 1) perception (CAM + MQTT) - để show dashboard / status
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

    dog = motion.dog

    # LED helper (pidog rgb_strip)
    def set_led(color: str, bps: float = 0.5):
        try:
            if dog and hasattr(dog, "rgb_strip"):
                dog.rgb_strip.set_mode("breath", color, bps=bps)
        except Exception:
            pass

    # ✅ boot default LED WHITE
    set_led("white", bps=0.4)

    # 3) active listening (upload + show transcript)
    listener = ActiveListener(
        mic_device=MIC_DEVICE,
        speaker_device=SPK_DEVICE,
        threshold=2500,
        cooldown_sec=1.0,
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        debug=False,
    )
    listener.start()

    # manual override
    def on_manual_cmd(move: str):
        manual["move"] = (move or "STOP").upper()
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    # build status for dashboard
    def build_status():
        st = planner.get_status_dict()
        st.update({
            "manual_move": manual["move"],
            "manual_active": bool(manual_active()),
            "listener_transcript": getattr(listener, "last_transcript", None),
            "listener_label": getattr(listener, "last_label", None),
            "mic_device": MIC_DEVICE,
            "spk_device": SPK_DEVICE,
            "flow_wait_idle_sec": WAIT_IDLE_SEC,
            "action_sec": ACTION_SEC,
        })
        return json_safe(st)

    # 4) web dashboard
    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_jpeg=lambda: planner.latest_jpeg,
        get_status=build_status,
        get_minimap_png=planner.get_mini_map_png,
        on_manual_cmd=on_manual_cmd,
    )
    threading.Thread(target=web.run, daemon=True).start()

    # ===== Helpers =====
    def apply_pose_config():
        try:
            cfg = motion.load_pose_config()
            motion.apply_pose_from_cfg(cfg, per_servo_delay=0.02, settle_sec=0.6)
        except Exception:
            pass

    def do_stand():
        try:
            motion.execute("STOP")
        except Exception:
            pass

    def do_action_for(action: str, sec: float):
        """Chạy action trong sec giây, nhưng nếu manual thì thoát ngay."""
        t0 = time.time()
        while time.time() - t0 < sec:
            if manual_active():
                return
            try:
                motion.execute(action)
            except Exception:
                pass
            time.sleep(0.02)

    # ===== Random actions =====
    ACTIONS = [
        "FORWARD",      # đi tới
        "BACKWARD",     # lùi
        "TURN_LEFT",    # xoay trái
        "TURN_RIGHT",   # xoay phải
        "STOP",         # đứng yên
    ]

    def pick_random_action():
        return random.choice(ACTIONS)

    # ===== Flow State =====
    state = "IDLE_WAIT"     # IDLE_WAIT -> APPLY_POSE -> DO_RANDOM -> STAND_AFTER
    until = time.time() + WAIT_IDLE_SEC
    chosen_action = "STOP"

    try:
        # đứng yên lúc bắt đầu
        do_stand()
        set_face("what_is_it")

        while True:
            now = time.time()

            # 1) manual override ưu tiên
            if manual_active():
                set_led("white", bps=0.6)
                set_face("music")  # manual đang điều khiển => mặt vui (tuỳ bạn đổi)
                motion.execute(manual["move"])
                time.sleep(0.02)
                continue
            else:
                manual["move"] = None

            # 2) flow random
            if state == "IDLE_WAIT":
                do_stand()
                set_face("what_is_it")
                if now >= until:
                    state = "APPLY_POSE"

            elif state == "APPLY_POSE":
                # về pose config trước khi random
                set_face("what_is_it")
                apply_pose_config()

                chosen_action = pick_random_action()
                state = "DO_RANDOM"
                until = time.time() + ACTION_SEC

                # đổi mặt theo action
                if chosen_action == "STOP":
                    set_face("sleep")
                elif chosen_action == "BACKWARD":
                    set_face("sad")
                elif chosen_action in ("TURN_LEFT", "TURN_RIGHT"):
                    set_face("suprise")
                else:  # FORWARD
                    set_face("music")

            elif state == "DO_RANDOM":
                # chạy action trong khoảng ACTION_SEC
                if chosen_action == "STOP":
                    do_stand()
                else:
                    do_action_for(chosen_action, 0.10)  # gọi liên tục từng lát nhỏ

                if now >= until:
                    state = "STAND_AFTER"
                    until = time.time() + 0.25

            elif state == "STAND_AFTER":
                do_stand()
                set_face("what_is_it")
                if now >= until:
                    state = "IDLE_WAIT"
                    until = time.time() + WAIT_IDLE_SEC

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")

    finally:
        try:
            set_led("white", bps=0.2)
        except Exception:
            pass

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


if __name__ == "__main__":
    main()
