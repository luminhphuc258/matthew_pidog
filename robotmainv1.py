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

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

MIC_DEVICE = "default"
SPK_DEVICE = "default"

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

SAFE_DIST_CM = 50.0
EMERGENCY_STOP_CM = 10.0

STAND_HOLD_SEC = 3.0
TURN_RIGHT_SEC = 5.0


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


def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")

    # 1) perception
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
    # ✅ enable_head_wiggle=False -> P10 không lắc (nhưng P8/P9 vẫn giữ lực theo patch bạn sửa)
    motion = MotionController(pose_file=POSE_FILE, enable_head_wiggle=False)
    motion.boot()
    set_volumes()
    dog = motion.dog

    # LED via pidog
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

    # 5) UART reader (direct)
    sensor = UartSensorReader(SERIAL_PORT, BAUD_RATE)
    sensor.start()

    # manual override
    manual = {"move": None, "ts": 0.0}
    manual_timeout_sec = 1.2

    def on_manual_cmd(move: str):
        manual["move"] = move
        manual["ts"] = time.time()

    def manual_active():
        return manual["move"] and (time.time() - manual["ts"] < manual_timeout_sec)

    # special sequence state
    seq = {"state": "NONE", "until": 0.0}  # NONE | STAND | TURN_RIGHT

    def block_density(sectors):
        n = len(sectors)
        c = n // 2
        center3 = [sectors[c-1], sectors[c], sectors[c+1]] if n >= 3 else sectors
        center_blocked = sum(1 for s in center3 if s == "blocked")
        total_blocked = sum(1 for s in sectors if s == "blocked")
        return center_blocked, total_blocked

    def status_payload():
        st = planner.get_status_dict()
        st.update(sensor.snapshot())
        st.update({
            "manual_move": manual["move"],
            "manual_active": bool(manual_active()),
            "listener_transcript": listener.last_transcript,
            "listener_label": listener.last_label,
            "seq_state": seq["state"],
            "seq_until": seq["until"],
            "mic_device": MIC_DEVICE,
            "spk_device": SPK_DEVICE,
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

    try:
        set_led("breath", "white", bps=0.4)

        while True:
            st = planner.get_state()
            s = sensor.snapshot()
            dist = s.get("uart_dist_cm")

            decision = st.decision

            if manual_active():
                decision = manual["move"]
            else:
                manual["move"] = None

            center_blocked, _ = block_density(st.sector_states)
            now = time.time()

            # emergency by ultrasonic
            if dist is not None and dist < EMERGENCY_STOP_CM:
                decision = "STOP"

            trigger = (center_blocked >= 2) or (dist is not None and dist < SAFE_DIST_CM)

            # sequence: stand 3s -> turn right 5s
            if seq["state"] == "NONE":
                if (not manual_active()) and trigger:
                    seq["state"] = "STAND"
                    seq["until"] = now + STAND_HOLD_SEC
            elif seq["state"] == "STAND":
                if now >= seq["until"]:
                    seq["state"] = "TURN_RIGHT"
                    seq["until"] = now + TURN_RIGHT_SEC
            elif seq["state"] == "TURN_RIGHT":
                if now >= seq["until"]:
                    seq["state"] = "NONE"
                    seq["until"] = 0.0

            if not manual_active():
                if seq["state"] == "STAND":
                    decision = "STOP"
                elif seq["state"] == "TURN_RIGHT":
                    decision = "TURN_RIGHT"

            # LED
            danger = (seq["state"] != "NONE") or (center_blocked >= 2) or (dist is not None and dist < SAFE_DIST_CM)
            if danger:
                set_led("breath", "red", bps=0.8)
            else:
                if decision == "FORWARD":
                    set_led("breath", "blue", bps=0.6)
                elif decision in ("TURN_LEFT", "TURN_RIGHT"):
                    set_led("breath", "white", bps=0.6)
                else:
                    set_led("breath", "white", bps=0.4)

            # face
            if danger:
                face.set_face("angry")
            elif decision == "FORWARD":
                face.set_face("music")
            elif decision in ("TURN_LEFT", "TURN_RIGHT"):
                face.set_face("what_is_it")
            elif decision == "BACK":
                face.set_face("angry")
            else:
                face.set_face("what_is_it")

            motion.execute(decision)
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
