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


# ===================== utils =====================
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


def dict_get_any(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_ultrasonic_cm_from_state(st):
    if isinstance(st, dict):
        v = dict_get_any(st, ("uart_dist_cm", "dist_cm", "distance_cm", "distance", "dist"), None)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    # object-style
    for k in ("uart_dist_cm", "dist_cm", "distance_cm", "distance", "dist"):
        v = getattr(st, k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None


def get_decision_from_state(st) -> str:
    if isinstance(st, dict):
        return norm_move(st.get("decision", "FORWARD") or "FORWARD")
    return norm_move(getattr(st, "decision", "FORWARD") or "FORWARD")


def short(x, n=80):
    s = str(x)
    return s if len(s) <= n else s[:n] + "..."


def try_get_attr(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            v = getattr(obj, name, None)
            if v is not None:
                return v
    return default


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

    # ---------- 2) Motion: chỉ boot + stand ----------
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

    def do_stand_only():
        # đổi sang "STAND" nếu lib bạn có lệnh riêng
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

            # fps estimate mỗi ~1s
            if now - fps_t0 >= 1.0:
                fps_est = (frame_count - fps_n0) / max(1e-6, (now - fps_t0))
                fps_t0 = now
                fps_n0 = frame_count

            return frame
        except Exception:
            return None

    def get_ultrasonic_cm_from_planner():
        st = safe_get_state(planner)
        return get_ultrasonic_cm_from_state(st)

    # ---------- 4) AvoidObstacle ----------
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
        get_frame_bgr=get_frame_bgr_from_planner,
    )
    avoid.start()

    # ---------- 5) Manual/web ----------
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = (move or "STOP").upper()
        manual["ts"] = time.time()
        print(f"[MANUAL] cmd={manual['move']} (LOG ONLY)")

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    # ---------- 6) WebDashboard ----------
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

    # ---------- 7) logging helpers ----------
    def extract_planner_debug(st):
        """
        Trả về dict debug ngắn gọn: dist, decision, sector info nếu có.
        """
        out = {}
        if st is None:
            return out

        if isinstance(st, dict):
            out["decision"] = norm_move(st.get("decision", "UNKNOWN"))
            out["uart_dist_cm"] = dict_get_any(st, ("uart_dist_cm", "dist_cm", "distance_cm", "distance", "dist"), None)
            # vài key bạn hay có thể có (tùy PerceptionPlanner bạn viết)
            for k in ("safe", "emergency", "obstacle", "roi", "roi_score", "confidence", "min_dist_sector",
                      "sector_scores", "sector_dist", "sectors", "lidar_min_cm"):
                if k in st:
                    out[k] = st.get(k)
            # nếu có fps/cam status trong state
            for k in ("cam_fps", "cam_ok", "jpeg_ts", "frame_id"):
                if k in st:
                    out[k] = st.get(k)
            return out

        # object style
        out["decision"] = norm_move(getattr(st, "decision", "UNKNOWN"))
        out["uart_dist_cm"] = getattr(st, "uart_dist_cm", None)
        for k in ("safe", "emergency", "obstacle", "roi", "roi_score", "confidence", "min_dist_sector",
                  "sector_scores", "sector_dist", "sectors", "lidar_min_cm", "cam_fps", "cam_ok", "jpeg_ts", "frame_id"):
            v = getattr(st, k, None)
            if v is not None:
                out[k] = v
        return out

    def extract_avoid_debug(avoid_obj):
        """
        Cố gắng lấy internal status của AvoidObstacle (tùy bạn implement).
        """
        dbg = {}
        # plan string/dict
        plan = try_get_attr(avoid_obj, ("latest_plan", "plan", "current_plan", "last_plan"), None)
        dbg["plan"] = plan

        # thời gian plan / ttl / last trigger (tùy field bạn đặt)
        for name in ("plan_until_ts", "plan_expire_ts", "last_trigger_ts", "last_send_ts", "last_plan_ts"):
            v = getattr(avoid_obj, name, None)
            if v is not None:
                dbg[name] = v

        # nếu avoid lưu debug vision result / server result
        for name in ("last_server_resp", "last_resp", "last_result", "debug", "last_debug"):
            v = getattr(avoid_obj, name, None)
            if v is not None:
                dbg[name] = v

        return dbg

    def compute_would_move(planner_decision, dist_cm, avoid_dbg, manual_cmd=None):
        # manual ưu tiên (nhưng LOG ONLY)
        if manual_cmd:
            would = norm_move(manual_cmd)
        else:
            plan = avoid_dbg.get("plan", None)
            if isinstance(plan, str) and plan.strip():
                would = norm_move(plan)
            elif isinstance(plan, dict):
                # nếu plan dict có key action/decision
                a = plan.get("action", None) or plan.get("decision", None)
                would = norm_move(a) if a else planner_decision
            else:
                would = planner_decision

        # emergency stop
        try:
            if dist_cm is not None and float(dist_cm) < 10.0:
                return "STOP"
        except Exception:
            pass
        return would

    # ---------- start ----------
    set_face("what_is_it")
    do_stand_only()

    log_hz = 8.0
    log_dt = 1.0 / log_hz
    last_log = 0.0

    print("[START] Verbose Camera + Obstacle test mode")
    print("        - Robot STAND only (NO MOVE)")
    print("        - Planner + AvoidObstacle + WebDashboard running")
    print("        - Verbose logs: dist/decision/plan/FPS/frame_age/web/manual\n")

    try:
        while True:
            now = time.time()

            # luôn giữ đứng yên
            do_stand_only()

            # web toggles (nếu WebDashboard bạn có method này)
            try:
                auto_only = web.is_auto_on()
            except Exception:
                auto_only = None
            try:
                listen_only = web.is_listen_on()
            except Exception:
                listen_only = None
            try:
                listen_run = web.is_listen_run_on()
            except Exception:
                listen_run = None

            st = safe_get_state(planner)
            planner_decision = get_decision_from_state(st) if st is not None else "UNKNOWN"
            dist_cm = get_ultrasonic_cm_from_state(st)

            planner_dbg = extract_planner_debug(st)
            avoid_dbg = extract_avoid_debug(avoid)

            # frame health
            frame_age_ms = None
            if last_frame_ts > 0:
                frame_age_ms = (now - last_frame_ts) * 1000.0

            m_active = manual_active()
            m_cmd = manual["move"] if m_active else None

            would_move = compute_would_move(planner_decision, dist_cm, avoid_dbg, manual_cmd=m_cmd)

            if now - last_log >= log_dt:
                last_log = now

                # log 1: status ngắn gọn
                print(
                    f"[STAT] dist_cm={str(dist_cm):>6} | "
                    f"planner={planner_decision:<10} | "
                    f"avoid_plan={short(avoid_dbg.get('plan', None), 30):<33} | "
                    f"manual={'ON ' if m_active else 'OFF'}{(m_cmd or ''):<9} | "
                    f"web(auto/listen/run)={auto_only}/{listen_only}/{listen_run} | "
                    f"cam_fps~={fps_est:.1f} | frame_age_ms={('%.0f' % frame_age_ms) if frame_age_ms is not None else 'NA':>4} | "
                    f"would_move={would_move:<10} | EXEC=NO"
                )

                # log 2: planner debug (chọn vài key chính)
                # Bạn có thể thêm/bớt key ở đây cho gọn/đủ.
                keys_focus = ["uart_dist_cm", "safe", "emergency", "obstacle", "roi_score",
                              "min_dist_sector", "sector_scores", "sector_dist", "lidar_min_cm",
                              "confidence", "cam_fps", "cam_ok", "frame_id"]
                snap = {k: planner_dbg.get(k) for k in keys_focus if k in planner_dbg}
                if snap:
                    print(f"[PLAN] {short(snap, 220)}")

                # log 3: avoid debug (nếu có)
                akeys = ["plan_until_ts", "plan_expire_ts", "last_trigger_ts", "last_send_ts", "last_plan_ts",
                         "last_server_resp", "last_resp", "last_result", "last_debug"]
                asnap = {k: avoid_dbg.get(k) for k in akeys if k in avoid_dbg}
                if asnap:
                    print(f"[AVOD] {short(asnap, 220)}")

                # phân tách dễ nhìn
                print("-" * 120)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")

    finally:
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
