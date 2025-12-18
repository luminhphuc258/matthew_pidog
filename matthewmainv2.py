#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import subprocess
import socket
from pathlib import Path

import numpy as np
import cv2

from perception_planner import PerceptionPlanner
from motion_controller import MotionController
from web_dashboard import WebDashboard
from avoid_obstacle import AvoidObstacle, AvoidCfg

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

FACE_HOST = "127.0.0.1"
FACE_PORT = 39393
_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

BARK_CANDIDATES = [
    str(Path(__file__).resolve().parent / "sounds" / "bark.wav"),
    "/home/pi/sounds/bark.wav",
    "/usr/share/sounds/alsa/Front_Center.wav",
]


def set_face(emo: str):
    try:
        _face_sock.sendto(emo.encode("utf-8"), (FACE_HOST, FACE_PORT))
    except Exception:
        pass


def run(cmd):
    return subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def set_volumes():
    run(["amixer", "-q", "sset", "robot-hat speaker", "100%"])
    run(["amixer", "-q", "sset", "robot-hat speaker Playback Volume", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic Capture Volume", "100%"])


def norm_move(m: str) -> str:
    m = (m or "STOP").upper()
    if m == "BACKWARD":
        return "BACK"
    return m


def safe_get_state(planner):
    try:
        return planner.get_state()
    except Exception:
        return None


def get_lidar_cm_from_state(st):
    if st is None:
        return None
    v = getattr(st, "uart_dist_cm", None)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def file_exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


def bark_twice():
    wav = None
    for p in BARK_CANDIDATES:
        if file_exists(p):
            wav = p
            break
    if not wav:
        return
    run(["aplay", "-q", wav])
    time.sleep(0.10)
    run(["aplay", "-q", wav])


def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # 1) Planner
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        safe_dist_cm=50.0,
        emergency_stop_cm=10.0,
        enable_camera=True,
        enable_imu=True,
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

    # 2) Motion
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()
    set_volumes()
    set_face("what_is_it")

    dog = getattr(motion, "dog", None)

    def led_off():
        try:
            if dog and hasattr(dog, "rgb_strip"):
                try:
                    dog.rgb_strip.off()
                except Exception:
                    dog.rgb_strip.set_mode("solid", "black")
        except Exception:
            pass

    def led_blue():
        try:
            if dog and hasattr(dog, "rgb_strip"):
                dog.rgb_strip.set_mode("solid", "blue")
        except Exception:
            pass

    def led_red():
        try:
            if dog and hasattr(dog, "rgb_strip"):
                dog.rgb_strip.set_mode("solid", "red")
        except Exception:
            pass

    # 3) Frame provider
    def get_frame_bgr_from_planner():
        jpg = getattr(planner, "latest_jpeg", None)
        if not jpg:
            return None
        try:
            arr = np.frombuffer(jpg, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def get_lidar_cm_from_planner():
        st = safe_get_state(planner)
        return get_lidar_cm_from_state(st)

    # 4) AvoidObstacle
    avoid_cfg = AvoidCfg(
        loop_hz=15.0,
        roi_y_start_ratio=0.55,
        roi_y_end_ratio=1.0,
        sector_n=9,

        force_stop_cm=40.0,
        trigger_cm=120.0,
        hard_stop_cm=28.0,

        server_url="https://embeddedprogramming-healtheworldserver.up.railway.app/avoid_obstacle_vision",
        send_w=256,
        send_h=144,
        jpeg_quality=55,
        min_trigger_interval_sec=1.2,
        plan_ttl_sec=6.0,

        bands_n=6,
        corridor_min_width_ratio=0.18,
    )

    avoid = AvoidObstacle(
        get_distance_cm=get_lidar_cm_from_planner,
        cfg=avoid_cfg,
        get_frame_bgr=get_frame_bgr_from_planner,
        rotate180=True,
    )
    avoid.start()

    # 5) WebDashboard + manual
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = norm_move(move or "STOP")
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

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

    # 6) IMU bump handling (dựa vào PerceptionState.imu_bump)
    bump_score = 0.0
    bump_threshold = 2.4
    last_bump_t = time.time()

    def update_bump_score():
        nonlocal bump_score, last_bump_t
        st = safe_get_state(planner)
        now = time.time()
        dt = max(1e-3, now - last_bump_t)
        last_bump_t = now

        # decay
        bump_score *= (0.85 ** dt)

        if st is None:
            return

        try:
            if bool(getattr(st, "imu_bump", False)):
                bump_score += 1.2
        except Exception:
            pass

    # 7) Motion helpers
    last_cmd = "STOP"
    last_cmd_ts = 0.0

    def exec_move(cmd: str):
        nonlocal last_cmd, last_cmd_ts
        cmd = norm_move(cmd)
        now = time.time()

        if cmd == last_cmd and (now - last_cmd_ts) < 0.25:
            return
        last_cmd = cmd
        last_cmd_ts = now

        try:
            motion.execute(cmd)
        except Exception:
            pass

        if cmd == "FORWARD":
            led_blue()
        elif cmd == "BACK":
            led_red()
        else:
            led_off()

    def rotate_180_for_4s():
        exec_move("TURN_LEFT")
        t0 = time.time()
        while time.time() - t0 < 4.0:
            time.sleep(0.05)
        exec_move("STOP")

    def decide_auto() -> str:
        st = avoid.get_state()
        local = st.get("local", {}) if isinstance(st, dict) else {}
        plan = st.get("plan", {}) if isinstance(st, dict) else {}

        best = plan.get("best_sector", None)
        if not isinstance(best, int):
            best = local.get("best_sector", None)

        if isinstance(best, int):
            if best <= 3:
                return "TURN_LEFT"
            if best >= 5:
                return "TURN_RIGHT"
            return "FORWARD"
        return "FORWARD"

    last_bark_ts = 0.0
    bark_cooldown = 2.0

    print("[START] AUTO MOVE + LiDAR priority + IMU bump stop/rotate + GPT walkway/labels")

    try:
        while True:
            now = time.time()

            update_bump_score()

            if manual_active():
                exec_move(manual["move"])
                time.sleep(0.05)
                continue

            if not web.is_auto_on():
                exec_move("STOP")
                time.sleep(0.08)
                continue

            dist_cm = get_lidar_cm_from_planner()

            # IMU bump repeated
            if bump_score >= bump_threshold:
                exec_move("STOP")
                rotate_180_for_4s()
                avoid.request_plan_now(reason="imu_bump_rotate", force=True)
                bump_score = 0.0
                time.sleep(0.12)
                continue

            # LiDAR < 40 => STOP + bark + force GPT
            if dist_cm is not None and dist_cm < avoid_cfg.force_stop_cm:
                exec_move("STOP")
                if now - last_bark_ts >= bark_cooldown:
                    last_bark_ts = now
                    threading.Thread(target=bark_twice, daemon=True).start()
                avoid.request_plan_now(reason="lidar_force_stop", force=True)
                time.sleep(0.12)
                continue

            # narrow/no path => rotate then replan
            if avoid.should_rotate_replan():
                exec_move("STOP")
                rotate_180_for_4s()
                avoid.request_plan_now(reason="no_path_rotate", force=True)
                time.sleep(0.12)
                continue

            cmd = decide_auto()

            # extra safety: if close then don't forward hard
            if dist_cm is not None and dist_cm < 60.0 and cmd == "FORWARD":
                cmd = "TURN_LEFT"

            exec_move(cmd)
            time.sleep(0.06)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")

    finally:
        try:
            exec_move("STOP")
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
