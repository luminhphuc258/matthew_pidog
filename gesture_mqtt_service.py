#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import socket
import ssl
import threading

import cv2
import paho.mqtt.client as mqtt

from handcommand import HandCommand, HandCfg
from web_dashboard import WebDashboard


# ===== face UDP (giống face3d) =====
FACE_HOST = "127.0.0.1"
FACE_PORT = 39393
_face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def set_face(emo: str):
    try:
        _face_sock.sendto(emo.encode("utf-8"), (FACE_HOST, FACE_PORT))
    except Exception:
        pass


# ===== MQTT CONFIG (TLS insecure - no CA) =====
MQTT_HOST = "rfff7184.ala.us-east-1.emqxsl.com"
MQTT_PORT = 8883
MQTT_USER = "robot_matthew"
MQTT_PASS = "29061992abCD!yesokmen"

TOPIC_MAP = {
    "STOPMUSIC": "/robot/gesture/stopmusic",
    "STANDUP":   "robot/gesture/standup",
    "SIT":       "robot/gesture/sit",
    "MOVELEFT":  "robot/gesture/moveleft",
    "MOVERIGHT": "robot/moveright",
    "STOP":      "/robot/gesture/stop",
}

PUBLISH_COOLDOWN_SEC = 0.35   # chống spam


# ===== Snapshot export =====
SNAPSHOT_PATH = "/tmp/gesture_latest.jpg"
SNAPSHOT_JPEG_QUALITY = 80

# ✅ throttle snapshot mỗi ~5 giây
SNAPSHOT_INTERVAL_SEC = 5.0


class MqttPublisher:
    def __init__(self):
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)

        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self._connected = False
        self._lock = threading.Lock()
        self._last_pub = {}  # topic -> ts

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = (rc == 0)
        print(f"[MQTT] connected rc={rc}", flush=True)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[MQTT] disconnected rc={rc}", flush=True)

    def start(self):
        self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.client.loop_start()

    def stop(self):
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass

    def publish_action(self, action: str):
        topic = TOPIC_MAP.get(action)
        if not topic:
            return

        now = time.time()
        with self._lock:
            last = self._last_pub.get(topic, 0.0)
            if (now - last) < PUBLISH_COOLDOWN_SEC:
                return
            self._last_pub[topic] = now

        payload = b""
        try:
            self.client.publish(topic, payload=payload, qos=0, retain=False)
            print(f"[MQTT] publish {topic} ({action})", flush=True)
        except Exception as e:
            print("[MQTT] publish error:", e, flush=True)


def _write_snapshot_atomic(frame_bgr, out_path: str, quality: int = 80):
    """
    Ghi JPEG ra file theo kiểu atomic để process khác đọc không bị file hỏng.
    """
    try:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp_path, out_path)
    except Exception:
        pass


# ===== camera (shared) =====
class Camera:
    def __init__(self, dev="/dev/video0", w=640, h=480, fps=30,
                 snapshot_path: str = SNAPSHOT_PATH,
                 snapshot_interval_sec: float = SNAPSHOT_INTERVAL_SEC,
                 snapshot_quality: int = SNAPSHOT_JPEG_QUALITY):
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.last = None
        self.ts = 0.0

        self.snapshot_path = snapshot_path
        self.snapshot_interval_sec = float(snapshot_interval_sec)
        self.snapshot_quality = int(snapshot_quality)
        self._last_snapshot_ts = 0.0

        try:
            os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        except Exception:
            pass

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last = frame
        self.ts = time.time()

        # ✅ throttle snapshot mỗi N giây
        now = self.ts
        if (now - self._last_snapshot_ts) >= self.snapshot_interval_sec:
            self._last_snapshot_ts = now
            _write_snapshot_atomic(self.last, self.snapshot_path, quality=self.snapshot_quality)
            print(f"[SNAP] wrote {self.snapshot_path}", flush=True)

        return frame

    def get_frame(self):
        fr = self.read()
        return fr if fr is not None else self.last

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


def main():
    cam = Camera(dev="/dev/video0", w=640, h=480, fps=30,
                 snapshot_path=SNAPSHOT_PATH,
                 snapshot_interval_sec=SNAPSHOT_INTERVAL_SEC,
                 snapshot_quality=SNAPSHOT_JPEG_QUALITY)

    pub = MqttPublisher()
    pub.start()

    def on_action(action: str, face: str, bark: bool):
        print(f"[ACT ] {action}", flush=True)
        set_face(face)
        pub.publish_action(action)

    hc = HandCommand(
        cfg=HandCfg(
            cam_dev="/dev/video0",
            w=640, h=480, fps=30,
            process_every=2,
            action_cooldown_sec=0.7,

            pos_left_x=0.18,
            pos_right_x=0.82,
            pos_up_y=0.22,
            pos_down_y=0.78,
            pos_hold_frames=4,
        ),
        on_action=on_action,
        boot_helper=None,
        get_frame_bgr=cam.get_frame,
        open_own_camera=False,
        clear_memory_on_start=True
    )

    hc.start()
    hc.set_enabled(True)

    dash = WebDashboard(
        host="0.0.0.0",
        port=8000,
        get_frame_bgr=cam.get_frame,
        avoid_obstacle=None,
        on_manual_cmd=lambda m: print("[MANUAL CMD]", m, flush=True),
        rotate180=True,
        mqtt_enable=False,
        hand_command=hc,
    )

    print("\n=== Gesture MQTT Service + WebDashboard ===", flush=True)
    print("Snapshot:", SNAPSHOT_PATH, flush=True)
    print("Snapshot interval:", SNAPSHOT_INTERVAL_SEC, "sec", flush=True)
    print("Open browser: http://<pi_ip>:8000\n", flush=True)

    try:
        dash.run()
    finally:
        hc.stop()
        cam.close()
        pub.stop()


if __name__ == "__main__":
    main()
