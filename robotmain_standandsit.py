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
ACTION_SEC = 2.0

# ===== MANUAL OVERRIDE =====
manual = {"move": None, "ts": 0.0}
manual_timeout_sec = 1.2

# ========= FACE SERVICE (UDP) =========
FACE_HOST = "127.0.0.1"
FACE_PORT = 39393
_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def set_face(emo: str):
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

    # 1) perception (CAM + decision)
    # NOTE: nếu planner cần MQTT để quyết định né vật cản, giữ enable_mqtt=True như bạn đang dùng.
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

    dog = getattr(motion, "dog", None)

    # LED helper (pidog rgb_strip)
    def set_led(color: str, bps: float = 0.5):
        try:
            if dog and hasattr(dog, "rgb_strip"):
                dog.rgb_strip.set_mode("breath", color, bps=bps)
        except Exception:
            pass

    set_led("white", bps=0.4)

    # 3) Active listening (we will recreate this object when toggled ON)
    listener = None

    def start_listener():
        nonlocal listener
        if listener is not None:
            return
        listener = ActiveListener(
            mic_device=MIC_DEVICE,
            speaker_device=SPK_DEVICE,
            threshold=2500,
            cooldown_sec=1.0,
            nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
            debug=False,
        )
        listener.start()

    def stop_listener():
        nonlocal listener
        if listener is None:
            return
        try:
            listener.stop()
        except Exception:
            pass
        try:
            listener.join(2.0)
        except Exception:
            pass
        listener = None

    # default: listener ON lúc boot (bạn có thể đổi thành OFF nếu muốn tiết kiệm pin)
    start_listener()

    # manual override
    def on_manual_cmd(move: str):
        manual["move"] = (move or "STOP").upper()
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    # 4) web dashboard (MQTT trực tiếp ở WebDashboard)
    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_jpeg=lambda: planner.latest_jpeg,
        on_manual_cmd=on_manual_cmd,

        mqtt_enable=True,
        mqtt_host="rfff7184.ala.us-east-1.emqxsl.com",
        mqtt_port=8883,
        mqtt_user="robot_matthew",
        mqtt_pass="29061992abCD!yesokmen",
        mqtt_topic="/pidog/sensorhubdata",
        mqtt_client_id="pidog-webdash",
        mqtt_tls=True,       # EMQX SSL
        mqtt_insecure=True,
        mqtt_debug=False,
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

    # normalize command names
    def norm_move(m: str) -> str:
        m = (m or "STOP").upper()
        if m == "BACKWARD":
            return "BACK"  # nếu MotionController của bạn dùng BACK
        return m

    # ===== Random actions =====
    ACTIONS = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]

    def pick_random_action():
        return random.choice(ACTIONS)

    # ===== Flow State =====
    state = "IDLE_WAIT"
    until = time.time() + WAIT_IDLE_SEC
    chosen_action = "STOP"

    # ===== Mode cache (để tránh stop/start listener liên tục) =====
    last_listen_on = None

    try:
        do_stand()
        set_face("what_is_it")

        while True:
            now = time.time()

            # ===== 0) đọc toggles từ web =====
            listen_on = web.is_listen_on()
            auto_on = web.is_auto_on()

            # ===== 1) LISTENING MODE: ưu tiên cao nhất -> robot đứng yên tuyệt đối =====
            if listen_on:
                # tắt auto move (WebDashboard đã auto tắt khi bật listening, nhưng mình vẫn chặn ở đây)
                # tắt motion hoàn toàn
                do_stand()
                set_led("white", bps=0.25)
                set_face("sleep")

                # tắt listener thật để tiết kiệm pin (nếu đang chạy)
                if last_listen_on is not True:
                    start_listener()
                time.sleep(0.05)
                last_listen_on = True
                continue
            else:
                # listening OFF -> tắt listener để tiết kiệm pin
                if last_listen_on is not False:
                    stop_listener()
                last_listen_on = False

            # ===== 2) MANUAL OVERRIDE (chỉ khi không Listening) =====
            if manual_active():
                set_led("white", bps=0.6)
                set_face("music")
                motion.execute(norm_move(manual["move"]))
                time.sleep(0.02)
                continue
            else:
                manual["move"] = None

            # ===== 3) AUTO MOVE MODE (chỉ khi không Listening) =====
            if auto_on:
                st = planner.get_state()
                decision = norm_move(getattr(st, "decision", "FORWARD") or "FORWARD")

                # hard safety stop (ưu tiên)
                dist = getattr(st, "uart_dist_cm", None)
                try:
                    if dist is not None and float(dist) < 10.0:
                        decision = "STOP"
                except Exception:
                    pass

                # face + led
                if decision == "FORWARD":
                    set_face("music")
                    set_led("white", bps=0.5)
                elif decision in ("TURN_LEFT", "TURN_RIGHT"):
                    set_face("suprise")
                    set_led("white", bps=0.8)
                elif decision in ("BACK", "BACKWARD"):
                    set_face("sad")
                    set_led("white", bps=0.8)
                else:
                    set_face("what_is_it")
                    set_led("white", bps=0.3)

                motion.execute(decision)
                time.sleep(0.02)
                continue

            # ===== 4) RANDOM FLOW (khi Auto Move OFF) =====
            if state == "IDLE_WAIT":
                do_stand()
                set_face("what_is_it")
                set_led("white", bps=0.25)
                if now >= until:
                    state = "APPLY_POSE"

            elif state == "APPLY_POSE":
                set_face("what_is_it")
                apply_pose_config()

                chosen_action = pick_random_action()
                state = "DO_RANDOM"
                until = time.time() + ACTION_SEC

                # face by action
                if chosen_action == "STOP":
                    set_face("sleep")
                elif chosen_action == "BACKWARD":
                    set_face("sad")
                elif chosen_action in ("TURN_LEFT", "TURN_RIGHT"):
                    set_face("suprise")
                else:
                    set_face("music")

            elif state == "DO_RANDOM":
                action = norm_move(chosen_action)
                if action == "STOP":
                    do_stand()
                else:
                    do_action_for(action, 0.10)

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
            stop_listener()
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
