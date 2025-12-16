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
import cv2  # NEW

from perception_planner import PerceptionPlanner
from motion_controller import MotionController
from web_dashboard import WebDashboard
from active_listener import ActiveListener

# NEW: avoid obstacle
from avoid_obstacle import AvoidObstacle, AvoidCfg

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

MIC_DEVICE = "default"
SPK_DEVICE = "default"

WAIT_IDLE_SEC = 9.0
ACTION_SEC = 2.0

manual = {"move": None, "ts": 0.0}
manual_timeout_sec = 1.2

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
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # 1) perception
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

    def set_led(color: str, bps: float = 0.5):
        try:
            if dog and hasattr(dog, "rgb_strip"):
                dog.rgb_strip.set_mode("breath", color, bps=bps)
        except Exception:
            pass

    set_led("white", bps=0.4)

    # 3) active listener (create/stop on demand)
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

    def listener_is_playing() -> bool:
        try:
            return (listener is not None) and listener._playing.is_set()
        except Exception:
            return False

    # manual
    def on_manual_cmd(move: str):
        manual["move"] = (move or "STOP").upper()
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    # ========= NEW: frame provider for dashboard + avoid_obstacle =========
    def get_frame_bgr_from_planner():
        """
        planner.latest_jpeg -> decode -> BGR
        (để WebDashboard overlay + AvoidObstacle dùng chung 1 nguồn camera)
        """
        jpg = getattr(planner, "latest_jpeg", None)
        if not jpg:
            return None
        try:
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def get_ultrasonic_cm_from_planner():
        """
        Lấy dist từ planner.get_state() (dict hoặc object).
        Ưu tiên uart_dist_cm.
        """
        try:
            st = planner.get_state()
        except Exception:
            st = None

        # dict-style
        if isinstance(st, dict):
            for k in ("uart_dist_cm", "dist_cm", "distance_cm", "distance", "dist"):
                if k in st and st[k] is not None:
                    try:
                        return float(st[k])
                    except Exception:
                        return None
            return None

        # object-style fallback
        for k in ("uart_dist_cm", "dist_cm", "distance_cm", "distance", "dist"):
            v = getattr(st, k, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    # ========= NEW: AvoidObstacle =========
    avoid_cfg = AvoidCfg(
        loop_hz=15.0,
        roi_y_start_ratio=0.60,
        sector_n=9,
        trigger_cm=120.0,
        server_url="https://embeddedprogramming-healtheworldserver.up.railway.app/avoid_obstacle_vision",
        send_w=256,
        send_h=144,
        jpeg_quality=55,
        min_trigger_interval_sec=2.0,
        plan_ttl_sec=8.0,
    )

    avoid = AvoidObstacle(
        get_ultrasonic_cm=get_ultrasonic_cm_from_planner,
        cfg=avoid_cfg,
        get_frame_bgr=get_frame_bgr_from_planner,  # IMPORTANT: no extra camera
    )
    avoid.start()

    # 4) web dashboard (3 nút) — NEW: dùng get_frame_bgr + avoid_obstacle
    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_frame_bgr=get_frame_bgr_from_planner,
        avoid_obstacle=avoid,
        on_manual_cmd=on_manual_cmd,

        mqtt_enable=True,
        mqtt_host="rfff7184.ala.us-east-1.emqxsl.com",
        mqtt_port=8883,
        mqtt_user="robot_matthew",
        mqtt_pass="29061992abCD!yesokmen",
        mqtt_topic="/pidog/sensorhubdata",
        mqtt_client_id="pidog-webdash",
        mqtt_tls=True,
        mqtt_insecure=True,
        mqtt_debug=False,
    )
    threading.Thread(target=web.run, daemon=True).start()

    # helpers
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
        t0 = time.time()
        while time.time() - t0 < sec:
            if manual_active():
                return
            try:
                motion.execute(action)
            except Exception:
                pass
            time.sleep(0.02)

    def norm_move(m: str) -> str:
        m = (m or "STOP").upper()
        if m == "BACKWARD":
            return "BACK"
        return m

    ACTIONS = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]

    def pick_random_action():
        return random.choice(ACTIONS)

    state = "IDLE_WAIT"
    until = time.time() + WAIT_IDLE_SEC
    chosen_action = "STOP"

    try:
        do_stand()
        set_face("what_is_it")

        while True:
            now = time.time()

            listen_only = web.is_listen_on()
            auto_only = web.is_auto_on()
            listen_run = web.is_listen_run_on()

            # ===== MODE 1: Listen & Run (TEST) =====
            if listen_run:
                start_listener()
                set_face("music" if listener_is_playing() else "suprise")

                # decision lấy từ planner (dict hoặc object)
                st = None
                try:
                    st = planner.get_state()
                except Exception:
                    st = None

                if isinstance(st, dict):
                    decision = norm_move(st.get("decision", "FORWARD") or "FORWARD")
                    dist = st.get("uart_dist_cm", None)
                else:
                    decision = norm_move(getattr(st, "decision", "FORWARD") or "FORWARD")
                    dist = getattr(st, "uart_dist_cm", None)

                try:
                    if dist is not None and float(dist) < 10.0:
                        decision = "STOP"
                except Exception:
                    pass

                motion.execute(decision)
                time.sleep(0.02)
                continue

            # ===== MODE 2: Active Listening (listen-only) =====
            if listen_only:
                start_listener()
                do_stand()
                set_face("music" if listener_is_playing() else "suprise")
                time.sleep(0.05)
                continue

            # ===== MODE 3: Auto Move (move-only) =====
            if auto_only:
                stop_listener()

                st = None
                try:
                    st = planner.get_state()
                except Exception:
                    st = None

                if isinstance(st, dict):
                    decision = norm_move(st.get("decision", "FORWARD") or "FORWARD")
                    dist = st.get("uart_dist_cm", None)
                else:
                    decision = norm_move(getattr(st, "decision", "FORWARD") or "FORWARD")
                    dist = getattr(st, "uart_dist_cm", None)

                try:
                    if dist is not None and float(dist) < 10.0:
                        decision = "STOP"
                except Exception:
                    pass

                set_face("what_is_it")
                motion.execute(decision)
                time.sleep(0.02)
                continue

            # ===== NORMAL MODE: random flow + manual =====
            stop_listener()

            if manual_active():
                set_face("music")
                motion.execute(norm_move(manual["move"]))
                time.sleep(0.02)
                continue
            else:
                manual["move"] = None

            if state == "IDLE_WAIT":
                do_stand()
                set_face("what_is_it")
                if now >= until:
                    state = "APPLY_POSE"

            elif state == "APPLY_POSE":
                set_face("what_is_it")
                apply_pose_config()

                chosen_action = pick_random_action()
                state = "DO_RANDOM"
                until = time.time() + ACTION_SEC

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
            stop_listener()
        except Exception:
            pass
        try:
            avoid.stop()
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
