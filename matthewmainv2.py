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

# ===== Sound files (adjust if you want) =====
BARK_CANDIDATES = [
    str(Path(__file__).resolve().parent / "sounds" / "bark.wav"),
    "/home/pi/sounds/bark.wav",
    "/usr/share/sounds/alsa/Front_Center.wav",  # fallback
]

# ===================== utils =====================
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
    if m == "FORW":
        return "FORWARD"
    return m

def dict_get_any(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def safe_get_state(planner):
    try:
        return planner.get_state()
    except Exception:
        return None

def get_lidar_cm_from_state(st):
    # mqtt payload now: uart_dist_cm, uart_strength, temp_c, humid
    if isinstance(st, dict):
        v = dict_get_any(st, ("uart_dist_cm", "lidar_cm", "dist_cm", "distance_cm", "distance", "dist"), None)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    for k in ("uart_dist_cm", "lidar_cm", "dist_cm", "distance_cm", "distance", "dist"):
        v = getattr(st, k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None

def get_lidar_strength_from_state(st):
    if isinstance(st, dict):
        v = dict_get_any(st, ("uart_strength", "lidar_strength", "strength"), None)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    for k in ("uart_strength", "lidar_strength", "strength"):
        v = getattr(st, k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None

def file_exists(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False

def play_wav(path: str):
    # blocking play
    # aplay works well on Pi
    run(["aplay", "-q", path])

def bark_twice():
    wav = None
    for p in BARK_CANDIDATES:
        if file_exists(p):
            wav = p
            break
    if not wav:
        return
    play_wav(wav)
    time.sleep(0.08)
    play_wav(wav)


# ===================== main =====================
def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # ---------- 1) PerceptionPlanner (camera + MQTT sensor state) ----------
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

    # ---------- 2) MotionController ----------
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()
    set_volumes()

    dog = getattr(motion, "dog", None)

    # LED helpers (Robot HAT rgb_strip)
    def led_off():
        try:
            if dog and hasattr(dog, "rgb_strip"):
                # Many libs support "off" or set_mode("solid","black")
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

    # start pose
    set_face("what_is_it")
    try:
        motion.execute("STOP")
    except Exception:
        pass

    # ---------- 3) frame provider: planner.latest_jpeg -> BGR ----------
    last_frame_ts = 0.0
    frame_count = 0
    fps_est = 0.0
    fps_t0 = time.time()
    fps_n0 = 0

    def get_frame_bgr_from_planner():
        nonlocal last_frame_ts, frame_count, fps_est, fps_t0, fps_n0
        jpg = getattr(planner, "latest_jpeg", None)
        if not jpg:
            return None
        try:
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None

            now = time.time()
            last_frame_ts = now
            frame_count += 1
            if now - fps_t0 >= 1.0:
                fps_est = (frame_count - fps_n0) / max(1e-6, (now - fps_t0))
                fps_t0 = now
                fps_n0 = frame_count
            return frame
        except Exception:
            return None

    def get_lidar_cm_from_planner():
        st = safe_get_state(planner)
        return get_lidar_cm_from_state(st)

    def get_strength_from_planner():
        st = safe_get_state(planner)
        return get_lidar_strength_from_state(st)

    # ---------- 4) AvoidObstacle ----------
    avoid_cfg = AvoidCfg(
        loop_hz=15.0,
        roi_y_start_ratio=0.60,
        roi_y_end_ratio=1.00,
        sector_n=9,

        trigger_cm=120.0,
        hard_stop_cm=35.0,
        max_valid_cm=800.0,

        server_url="https://embeddedprogramming-healtheworldserver.up.railway.app/avoid_obstacle_vision",
        send_w=256,
        send_h=144,
        jpeg_quality=55,
        min_trigger_interval_sec=1.2,
        plan_ttl_sec=8.0,
    )

    avoid = AvoidObstacle(
        get_ultrasonic_cm=get_lidar_cm_from_planner,   # reuse signature: distance getter
        get_lidar_strength=get_strength_from_planner,
        cfg=avoid_cfg,
        get_frame_bgr=get_frame_bgr_from_planner,
        rotate180=True,
    )
    avoid.start()

    # ---------- 5) WebDashboard (manual override) ----------
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = norm_move(move or "STOP")
        manual["ts"] = time.time()
        print(f"[MANUAL] {manual['move']}")

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

    # ---------- 6) Decision helpers ----------
    def decide_from_avoid() -> str:
        """
        Prefer GPT plan best_sector (0..8). Fallback to local best_sector.
        Map:
          0..3  -> TURN_LEFT
          4     -> FORWARD
          5..8  -> TURN_RIGHT
        """
        st = avoid.get_state() if hasattr(avoid, "get_state") else {}
        plan = (st.get("plan") or {}) if isinstance(st, dict) else {}
        local = (st.get("local") or {}) if isinstance(st, dict) else {}

        best = None
        if isinstance(plan, dict):
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

    # bark debounce
    last_bark_ts = 0.0
    bark_cooldown = 2.0  # seconds

    # motion timing
    last_cmd = "STOP"
    last_cmd_ts = 0.0

    def exec_move(cmd: str):
        nonlocal last_cmd, last_cmd_ts
        cmd = norm_move(cmd)

        # reduce spam: only send if changed or older than 0.35s
        now = time.time()
        if cmd == last_cmd and (now - last_cmd_ts) < 0.35:
            return
        last_cmd = cmd
        last_cmd_ts = now

        try:
            motion.execute(cmd)
        except Exception as e:
            print("[MOTION] execute error:", e)

        # LED policy
        if cmd == "FORWARD":
            led_blue()
        elif cmd == "BACK":
            led_red()
        else:
            led_off()

    # ---------- 7) MAIN LOOP ----------
    print("[START] AUTO MOVE + AVOID + BARK + LED")
    print(" - Auto navigate using AvoidObstacle best_sector")
    print(" - Bark twice when obstacle near (LiDAR <= trigger_cm)")
    print(" - LED blue forward / red back / off stop\n")

    try:
        while True:
            now = time.time()

            # 1) read lidar
            dist_cm = get_lidar_cm_from_planner()
            strength = get_strength_from_planner()

            # 2) safety: hard stop/back if too near
            if dist_cm is not None and dist_cm <= avoid_cfg.hard_stop_cm:
                # bark (debounced)
                if now - last_bark_ts >= bark_cooldown:
                    last_bark_ts = now
                    threading.Thread(target=bark_twice, daemon=True).start()

                # back off a bit then stop
                exec_move("BACK")
                time.sleep(0.60)
                exec_move("STOP")
                time.sleep(0.10)
                continue

            # 3) obstacle trigger bark (less strict)
            if dist_cm is not None and dist_cm <= avoid_cfg.trigger_cm:
                if now - last_bark_ts >= bark_cooldown:
                    last_bark_ts = now
                    threading.Thread(target=bark_twice, daemon=True).start()

            # 4) manual override (from WebDashboard) has priority
            if manual_active():
                exec_move(manual["move"])
                time.sleep(0.05)
                continue

            # 5) auto decision from avoid
            cmd = decide_from_avoid()

            # 6) execute
            exec_move(cmd)

            # 7) small log
            if int(now * 2) % 2 == 0:  # ~1Hz-ish without timer
                print(f"[AUTO] lidar_cm={dist_cm} str={strength} cmd={cmd} fps~={fps_est:.1f}")

            time.sleep(0.05)

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
