#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import subprocess
import socket
from pathlib import Path
from typing import Optional

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


# ===================== utils =====================
def set_face(emo: str):
    try:
        _face_sock.sendto(emo.encode("utf-8"), (FACE_HOST, FACE_PORT))
    except Exception:
        pass


def run(cmd):
    return subprocess.run(cmd, check=False)


def set_volumes():
    # Tuỳ máy bạn: có thể khác control name
    run(["amixer", "-q", "sset", "robot-hat speaker", "100%"])
    run(["amixer", "-q", "sset", "robot-hat speaker Playback Volume", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic Capture Volume", "100%"])


def norm_move(m: str) -> str:
    m = (m or "STOP").upper().strip()
    if m in ("BACKWARD", "REVERSE"):
        return "BACK"
    if m in ("TURNLEFT", "LEFT_TURN"):
        return "LEFT"
    if m in ("TURNRIGHT", "RIGHT_TURN"):
        return "RIGHT"
    if m in ("FORWARD", "FWD"):
        return "FORWARD"
    if m in ("STOP", "IDLE"):
        return "STOP"
    if m in ("LEFT", "RIGHT", "BACK"):
        return m
    return "STOP"


def dict_get_any(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_lidar_cm_from_state(st) -> Optional[float]:
    if st is None:
        return None
    if isinstance(st, dict):
        v = dict_get_any(st, ("uart_dist_cm", "dist_cm", "distance_cm", "distance", "dist"), None)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    # object style
    for k in ("uart_dist_cm", "dist_cm", "distance_cm", "distance", "dist"):
        v = getattr(st, k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None


def get_decision_from_state(st) -> str:
    if st is None:
        return "STOP"
    if isinstance(st, dict):
        return norm_move(st.get("decision", "STOP"))
    return norm_move(getattr(st, "decision", "STOP"))


# ===================== PiDog voice + LED helpers =====================
def pidog_voice(dog, action: str):
    """
    action in ["bark", "bark harder", "pant", "howling"]
    Try multiple APIs because PiDog versions differ.
    """
    if dog is None:
        return False

    a = (action or "").strip().lower()

    # 1) common: dog.speak("bark") / dog.speak(action)
    for fn in ("speak", "voice", "play_voice", "play_sound", "sound"):
        try:
            if hasattr(dog, fn):
                getattr(dog, fn)(a)
                return True
        except Exception:
            pass

    # 2) some libs: dog.do_action("bark") / dog.action("bark")
    for fn in ("do_action", "action", "do_voice", "do_voice_action", "voice_action", "do_emotion"):
        try:
            if hasattr(dog, fn):
                getattr(dog, fn)(a)
                return True
        except Exception:
            pass

    # 3) some libs expose list/dict VOICE_ACTIONS and a runner
    try:
        va = getattr(dog, "VOICE_ACTIONS", None) or getattr(dog, "voice_actions", None)
        if va and isinstance(va, (list, tuple)) and a in [x.lower() for x in va]:
            # try generic executor
            for fn in ("run", "play", "execute"):
                if hasattr(dog, fn):
                    getattr(dog, fn)(a)
                    return True
    except Exception:
        pass

    return False


def bark_twice(dog):
    # bark 2 tiếng, cách nhau 150ms
    ok1 = pidog_voice(dog, "bark")
    time.sleep(0.15)
    ok2 = pidog_voice(dog, "bark")
    return ok1 or ok2


def set_led_mode(dog, mode: str):
    """
    mode: "FORWARD" => blue, "BACK" => red, "STOP" => off
    """
    if dog is None:
        return
    try:
        strip = getattr(dog, "rgb_strip", None)
        if strip is None:
            return

        m = norm_move(mode)
        if m == "FORWARD":
            strip.set_mode("breath", "blue", bps=0.8)
        elif m == "BACK":
            strip.set_mode("breath", "red", bps=0.8)
        else:
            # tắt để tiết kiệm pin
            # tuỳ lib: có thể là "off" hoặc set_mode("solid","black")
            try:
                strip.set_mode("off")
            except Exception:
                try:
                    strip.set_mode("solid", "black")
                except Exception:
                    pass
    except Exception:
        pass


# ===================== motion exec wrapper =====================
def exec_move(motion: MotionController, move: str):
    mv = norm_move(move)
    try:
        motion.execute(mv)
    except Exception:
        # fallback: nếu lib bạn tên khác
        try:
            motion.run(mv)
        except Exception:
            pass
    return mv


# ===================== main =====================
def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # ---------- 1) PerceptionPlanner ----------
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

    # ---------- 2) Motion ----------
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()
    set_volumes()

    dog = getattr(motion, "dog", None)

    # đứng dậy trước
    try:
        exec_move(motion, "STOP")
    except Exception:
        pass

    # ---------- 3) frame provider: planner.latest_jpeg -> BGR ----------
    last_frame_ts = 0.0

    def get_frame_bgr_from_planner():
        nonlocal last_frame_ts
        jpg = getattr(planner, "latest_jpeg", None)
        if not jpg:
            return None
        try:
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            last_frame_ts = time.time()
            return frame
        except Exception:
            return None

    def get_lidar_cm_from_planner():
        try:
            st = planner.get_status_dict()  # dict
        except Exception:
            try:
                st = planner.get_state()
            except Exception:
                st = None
        return get_lidar_cm_from_state(st)

    # ---------- 4) AvoidObstacle ----------
    avoid_cfg = AvoidCfg(
        loop_hz=15.0,
        roi_y_start_ratio=0.60,
        roi_y_end_ratio=1.00,
        sector_n=9,
        trigger_cm=120.0,
        hard_stop_cm=35.0,
        server_url="https://embeddedprogramming-healtheworldserver.up.railway.app/avoid_obstacle_vision",
        send_w=256,
        send_h=144,
        jpeg_quality=55,
        min_trigger_interval_sec=1.0,
        plan_ttl_sec=8.0,
    )

    avoid = AvoidObstacle(
        get_ultrasonic_cm=get_lidar_cm_from_planner,   # (distance getter)
        cfg=avoid_cfg,
        get_frame_bgr=get_frame_bgr_from_planner,
        rotate180=True,  # mặc định rotate 180
    )
    avoid.start()

    # ---------- 5) WebDashboard ----------
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = norm_move(move)
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

    # ---------- 6) Decision state machine ----------
    danger_mode = False
    danger_enter_ts = 0.0
    barked = False
    back_until_ts = 0.0
    turn180_until_ts = 0.0

    last_led_mode = None

    def set_led_for_move(mv: str):
        nonlocal last_led_mode
        mv = norm_move(mv)
        if mv != last_led_mode:
            set_led_mode(dog, mv)
            last_led_mode = mv

    set_face("what_is_it")
    set_led_for_move("STOP")

    print("[START] Auto move + avoid obstacle (LiDAR priority)")

    try:
        while True:
            now = time.time()

            # web toggles
            try:
                auto_on = bool(web.is_auto_on())
            except Exception:
                auto_on = True  # default auto ON if method not found

            # manual priority (optional)
            m_cmd = manual["move"] if manual_active() else None
            if m_cmd:
                mv = exec_move(motion, m_cmd)
                set_led_for_move(mv)
                time.sleep(0.02)
                continue

            # if auto off -> stop + led off
            if not auto_on:
                mv = exec_move(motion, "STOP")
                set_led_for_move(mv)
                time.sleep(0.05)
                continue

            # get sensors
            try:
                st = planner.get_state()
            except Exception:
                st = None

            dist_cm = get_lidar_cm_from_state(st)
            planner_decision = get_decision_from_state(st)

            # ===== (1) LiDAR hard priority: <40cm =====
            if dist_cm is not None and dist_cm < 40.0:
                if not danger_mode:
                    danger_mode = True
                    danger_enter_ts = now
                    barked = False

                # STOP ngay
                mv = exec_move(motion, "STOP")
                set_led_for_move(mv)

                # bark 2 tiếng (chỉ 1 lần khi mới vào danger)
                if not barked:
                    barked = True
                    bark_twice(dog)

                # ép gọi GPT (avoid trigger)
                try:
                    avoid.force_trigger()
                except Exception:
                    pass

                # lùi ra 0.8s để thoát sát
                back_until_ts = max(back_until_ts, now + 0.8)

            # ===== (2) BACK window =====
            if now < back_until_ts:
                mv = exec_move(motion, "BACK")
                set_led_for_move(mv)
                time.sleep(0.02)
                continue

            # ===== (3) TURN 180 window =====
            if now < turn180_until_ts:
                mv = exec_move(motion, "LEFT")  # quay trái
                set_led_for_move(mv)
                time.sleep(0.02)
                continue
            elif turn180_until_ts > 0 and now >= turn180_until_ts:
                turn180_until_ts = 0.0
                danger_mode = False
                mv = exec_move(motion, "STOP")
                set_led_for_move(mv)
                time.sleep(0.05)
                continue

            # ===== (4) Use GPT plan from AvoidObstacle if available =====
            plan = None
            try:
                plan = avoid.get_best_action()
            except Exception:
                plan = None

            if danger_mode:
                # nếu danger mà chưa có plan rõ ràng -> đứng yên, quá 2s thì quay 180
                if (not plan) or (float(plan.get("confidence", 0.0) or 0.0) < 0.35):
                    mv = exec_move(motion, "STOP")
                    set_led_for_move(mv)
                    if now - danger_enter_ts > 2.0:
                        turn180_until_ts = max(turn180_until_ts, now + 4.0)
                    time.sleep(0.03)
                    continue

                # nếu plan báo no_path/narrow -> quay 180 luôn
                if bool(plan.get("no_path", False)) or bool(plan.get("narrow", False)):
                    turn180_until_ts = max(turn180_until_ts, now + 4.0)
                    mv = exec_move(motion, "STOP")
                    set_led_for_move(mv)
                    time.sleep(0.03)
                    continue

                # có action rõ ràng
                act = norm_move(plan.get("action", "STOP"))
                mv = exec_move(motion, act)
                set_led_for_move(mv)

                # nếu đã đi được và LiDAR thoáng -> thoát danger
                if act != "STOP" and (dist_cm is None or dist_cm >= 60.0):
                    danger_mode = False

                time.sleep(0.02)
                continue

            # ===== (5) Normal mode: prefer GPT if confident, else planner =====
            if plan and float(plan.get("confidence", 0.0) or 0.0) >= 0.45:
                act = norm_move(plan.get("action", "STOP"))
                mv = exec_move(motion, act)
            else:
                mv = exec_move(motion, planner_decision)

            set_led_for_move(mv)

            # tiết kiệm pin: nếu STOP thì đừng loop quá nhanh
            time.sleep(0.02 if mv != "STOP" else 0.06)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")

    finally:
        try:
            exec_move(motion, "STOP")
            set_led_for_move("STOP")
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
