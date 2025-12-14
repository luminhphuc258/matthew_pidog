#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict

import serial

from perception_planner import PerceptionPlanner
from motion_controller import MotionController
from face_display import FaceDisplay
from web_dashboard import WebDashboard
from active_listener import ActiveListener
from dogbehavior import DogBehavior

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

MIC_DEVICE = "default"
SPK_DEVICE = "default"

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

SAFE_DIST_CM = 50.0
EMERGENCY_STOP_CM = 10.0


def run(cmd):
    return subprocess.run(cmd, check=False)


def set_volumes():
    run(["amixer", "-q", "sset", "robot-hat speaker", "100%"])
    run(["amixer", "-q", "sset", "robot-hat speaker Playback Volume", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic", "100%"])
    run(["amixer", "-q", "sset", "robot-hat mic Capture Volume", "100%"])


class UartSensorReader:
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.ok = False
        self.error: Optional[str] = None
        self.last_line: Optional[str] = None
        self.last_rx_ts: Optional[float] = None

        self.temp_c: Optional[float] = None
        self.humid: Optional[float] = None
        self.dist_cm: Optional[float] = None

        self._t: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=1.0):
        if self._t:
            self._t.join(timeout=timeout)

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "uart_ok": self.ok,
                "uart_error": self.error,
                "uart_last_line": self.last_line,
                "uart_last_rx_ts": self.last_rx_ts,
                "uart_temp_c": self.temp_c,
                "uart_humid": self.humid,
                "uart_dist_cm": self.dist_cm,
            }

    def _loop(self):
        ser = None
        while not self._stop.is_set():
            if ser is None:
                try:
                    ser = serial.Serial(self.port, self.baud, timeout=1)
                    time.sleep(1.5)
                    try:
                        ser.reset_input_buffer()
                    except Exception:
                        pass
                    with self._lock:
                        self.ok = True
                        self.error = None
                    print(f"[UART] connected: {self.port} @ {self.baud}")
                except Exception as e:
                    with self._lock:
                        self.ok = False
                        self.error = f"open fail: {e}"
                    time.sleep(2.0)
                    continue

            try:
                raw = ser.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                with self._lock:
                    self.last_line = line
                    self.last_rx_ts = time.time()

                parts = line.split(",")
                if len(parts) != 4:
                    continue

                t = float(parts[1])
                h = float(parts[2])
                d = float(parts[3])

                with self._lock:
                    self.temp_c = t
                    self.humid = h
                    self.dist_cm = d
                    self.ok = True
                    self.error = None

            except Exception as e:
                with self._lock:
                    self.ok = False
                    self.error = f"read fail: {e}"
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(0.5)

        try:
            if ser:
                ser.close()
        except Exception:
            pass


def listener_is_recording(listener) -> bool:
    # cố gắng bắt nhiều tên biến khác nhau
    for k in ("is_recording", "recording", "mic_recording", "mic_busy", "listening"):
        v = getattr(listener, k, None)
        if isinstance(v, bool):
            return v
    return False


def listener_is_playing(listener) -> bool:
    for k in ("is_playing", "playing", "speaking", "speaker_busy", "tts_playing"):
        v = getattr(listener, k, None)
        if isinstance(v, bool):
            return v
    return False


def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # 1) perception (camera)
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        serial_port=SERIAL_PORT,
        baud=BAUD_RATE,
        safe_dist_cm=SAFE_DIST_CM,
        emergency_stop_cm=EMERGENCY_STOP_CM,
        enable_imu=False,
        enable_camera=True,
        uart_debug=False,
        uart_filter_window=5,
    )
    planner.start()

    # 2) motion
    motion = MotionController(pose_file=POSE_FILE, enable_head_wiggle=False)
    motion.boot()
    set_volumes()
    dog = motion.dog

    def set_led(mode: str, color: str, bps: float = 0.6):
        try:
            if dog and hasattr(dog, "rgb_strip"):
                dog.rgb_strip.set_mode(mode, color, bps=bps)
        except Exception:
            pass

    # 3) face
    face = FaceDisplay(default_face="what_is_it", fps=60, fullscreen=True)
    face.start()

    # 4) active listening
    listener = ActiveListener(
        mic_device=MIC_DEVICE,
        speaker_device=SPK_DEVICE,
        threshold=2500,
        cooldown_sec=1.0,
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        debug=False,
    )
    listener.start()

    # 5) UART reader
    sensor = UartSensorReader(SERIAL_PORT, BAUD_RATE)
    sensor.start()

    # 6) DogBehavior
    behavior = DogBehavior(
        safe_dist_cm=SAFE_DIST_CM,
        emergency_stop_cm=EMERGENCY_STOP_CM,
        rotate_sec=5.0,            # xoay 360 trong 5s
        walk_rest_every_sec=60.0,  # 1 phút
        rest_sit_sec=3.0,          # sit 3s
        after_avoid_cooldown_sec=1.2
    )

    # manual override
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = move
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    def status_payload():
        st = planner.get_status_dict()
        st.update(sensor.snapshot())
        st.update({
            "manual_move": manual["move"],
            "manual_active": bool(manual_active()),
            "listener_transcript": getattr(listener, "last_transcript", None),
            "listener_label": getattr(listener, "last_label", None),
            "listener_is_recording": listener_is_recording(listener),
            "listener_is_playing": listener_is_playing(listener),
            "behavior_state": behavior.state,
        })
        return st

    web = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_jpeg=lambda: planner.latest_jpeg,
        get_status=status_payload,
        get_minimap_png=planner.get_mini_map_png,
        on_manual_cmd=on_manual_cmd,
    )
    threading.Thread(target=web.run, daemon=True).start()

    # helper: SIT action (motion.execute không có SIT)
    def do_sit_blocking():
        try:
            if dog:
                dog.do_action("sit", speed=20)
                dog.wait_all_done()
        except Exception:
            pass

    # main loop
    try:
        set_led("breath", "white", bps=0.4)

        while True:
            st = planner.get_state()
            s = sensor.snapshot()
            dist = s.get("uart_dist_cm")

            # manual
            m_active = bool(manual_active())
            m_move = manual["move"] if m_active else None
            if not m_active:
                manual["move"] = None

            # listener states
            rec = listener_is_recording(listener)
            play = listener_is_playing(listener)

            # base camera decision (chỉ là input cho behavior; behavior sẽ ưu tiên sensor trước)
            camera_decision = st.decision

            out = behavior.update(
                dist_cm=dist,
                sector_states=st.sector_states,
                camera_decision=camera_decision,
                cam_blocked=st.cam_blocked,
                imu_bump=st.imu_bump,
                manual_active=m_active,
                manual_move=m_move,
                is_recording=rec,
                is_playing=play,
            )

            # LED + face
            set_led(out.led_mode, out.led_color, bps=out.led_bps)
            try:
                face.set_face(out.face)
            except Exception:
                pass

            # ACT
            if out.decision == "SIT":
                do_sit_blocking()
            else:
                motion.execute(out.decision)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    finally:
        try:
            set_led("breath", "white", bps=0.2)
        except Exception:
            pass

        try:
            listener.stop()
            listener.join(2.0)
        except Exception:
            pass

        try:
            sensor.stop()
            sensor.join(1.0)
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
