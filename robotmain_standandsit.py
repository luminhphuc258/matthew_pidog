#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import random
import subprocess
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
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # 1) perception (CAM + MQTT) - chỉ để hiển thị dashboard / sensor / camera
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

    # 3) face
    face = FaceDisplay(default_face="what_is_it", fps=60, fullscreen=True)
    face.start()

    # 4) active listening (chỉ để upload + show transcript, KHÔNG block movement nữa)
    listener = ActiveListener(
        mic_device=MIC_DEVICE,
        speaker_device=SPK_DEVICE,
        threshold=2500,
        cooldown_sec=1.0,
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        debug=False,
    )
    listener.start()

    # manual override (web)
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 2.0

    def on_manual_cmd(move: str):
        manual["move"] = (move or "STOP").upper()
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    def build_status():
        st = planner.get_status_dict()
        st.update({
            "manual_move": manual["move"],
            "manual_active": bool(manual_active()),
            "listener_transcript": getattr(listener, "last_transcript", None),
            "listener_label": getattr(listener, "last_label", None),
            "mic_device": MIC_DEVICE,
            "spk_device": SPK_DEVICE,
            "mode": "IDLE_STAND",
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

    # ===== IDLE BEHAVIOR SCHEDULE =====
    # thỉnh thoảng SIT flow: sit -> stop(sát sàn) -> apply pose -> stand
    sit_interval_sec = 60.0        # mỗi 1 phút (bạn chỉnh)
    sit_hold_sec = 3.0             # ngồi 3s
    stop_after_sit_sec = 1.0       # sau sit, STOP để sát sàn (bạn muốn)
    settle_pose_sec = 0.6          # chờ sau apply pose config

    # thỉnh thoảng push up
    pushup_interval_sec = 180.0    # mỗi 3 phút
    pushup_jitter_sec = 30.0       # random thêm cho tự nhiên

    next_sit_ts = time.time() + sit_interval_sec
    next_push_ts = time.time() + pushup_interval_sec + random.uniform(-pushup_jitter_sec, pushup_jitter_sec)

    def do_sit_flow():
        # 1) sit
        motion.execute("SIT")
        time.sleep(max(0.1, float(sit_hold_sec)))

        # 2) stop để robot “nằm sát sàn”
        motion.execute("STOP")
        time.sleep(max(0.1, float(stop_after_sit_sec)))

        # 3) về pose config
        try:
            cfg = motion.load_pose_config()
            motion.apply_pose_from_cfg(cfg, per_servo_delay=0.02, settle_sec=0.0)
        except Exception:
            pass
        time.sleep(max(0.1, float(settle_pose_sec)))

        # 4) stand
        motion.execute("STAND")

    # main loop (IDLE)
    try:
        # giữ stand ngay khi chạy
        motion.execute("STAND")

        while True:
            now = time.time()

            # ===== manual override từ web =====
            if manual_active():
                cmd = manual["move"]
                # nếu bấm STOP thì giữ stand nhẹ
                motion.execute(cmd)

                # face theo manual
                if cmd in ("FORWARD", "BACK", "TURN_LEFT", "TURN_RIGHT", "LEFT", "RIGHT"):
                    face.set_face("what_is_it")
                elif cmd == "STOP":
                    face.set_face("what_is_it")
                else:
                    face.set_face("what_is_it")

                time.sleep(0.03)
                continue
            else:
                manual["move"] = None

            # ===== idle actions =====
            if now >= next_sit_ts:
                face.set_face("sad")  # tuỳ bạn
                do_sit_flow()
                next_sit_ts = time.time() + sit_interval_sec
                face.set_face("what_is_it")

            elif now >= next_push_ts:
                face.set_face("music")
                motion.execute("PUSH_UP")
                motion.execute("STAND")
                next_push_ts = time.time() + pushup_interval_sec + random.uniform(-pushup_jitter_sec, pushup_jitter_sec)
                face.set_face("what_is_it")

            else:
                # bình thường chỉ đứng yên
                motion.execute("STOP")
                face.set_face("what_is_it")

            time.sleep(0.05)

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
