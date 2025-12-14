#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, threading, json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

# MQTT
try:
    import ssl
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None


@dataclass
class PerceptionState:
    ts: float
    sector_states: List[str]          # len=sector_n: "free" | "blocked" | "unknown"
    decision: str                    # "FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "BACK" | "STOP"
    reason: str

    # SensorHub from MQTT
    uart_dist_raw_cm: Optional[float] = None   # keep name for your old web UI
    uart_dist_cm: Optional[float] = None
    uart_temp_c: Optional[float] = None
    uart_humid: Optional[float] = None

    # mqtt health
    mqtt_ok: bool = False
    mqtt_error: Optional[str] = None
    mqtt_last_rx_ts: Optional[float] = None
    mqtt_dt_ms: Optional[int] = None

    # health flags
    imu_bump: bool = False
    cam_blocked: bool = False

    # extra
    mode: str = "AUTO"
    manual_override: Optional[str] = None


def _to_py(x):
    """Convert numpy scalars to python scalars to avoid JSON serialization bugs."""
    try:
        import numpy as _np
        if isinstance(x, (_np.bool_,)):
            return bool(x)
        if isinstance(x, (_np.integer,)):
            return int(x)
        if isinstance(x, (_np.floating,)):
            return float(x)
    except Exception:
        pass
    return x


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(_to_py(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(_to_py(v)) for v in obj]
    return _to_py(obj)


class PerceptionPlanner:
    """
    Camera obstacle detection (edge-density per sector) + SensorHub via MQTT.
    Decision priority:
      1) emergency stop by dist
      2) near avoidance by dist + camera side free
      3) normal navigation by camera
    """

    def __init__(
        self,
        cam_dev="/dev/video0",
        w=640,
        h=480,
        fps=30,
        sector_n=9,
        map_h=80,

        safe_dist_cm=50.0,
        emergency_stop_cm=10.0,

        enable_camera=True,
        enable_imu=False,   # kept for compatibility (not implemented)

        # MQTT
        enable_mqtt=True,
        mqtt_host="localhost",
        mqtt_port=8883,
        mqtt_user="",
        mqtt_pass="",
        mqtt_topic="/pidog/sensorhubdata",
        mqtt_client_id="pidog-perception-pi",
        mqtt_debug=False,
        mqtt_insecure=True,
    ):
        self.cam_dev, self.w, self.h, self.fps = cam_dev, int(w), int(h), int(fps)
        self.sector_n = int(sector_n)
        self.map_h = int(map_h)
        self.map_w = self.sector_n

        self.safe_dist_cm = float(safe_dist_cm)
        self.emergency_stop_cm = float(emergency_stop_cm)

        self.enable_camera = bool(enable_camera)
        self.enable_imu = bool(enable_imu)

        self.enable_mqtt = bool(enable_mqtt)
        self.mqtt_host = mqtt_host
        self.mqtt_port = int(mqtt_port)
        self.mqtt_user = mqtt_user
        self.mqtt_pass = mqtt_pass
        self.mqtt_topic = mqtt_topic
        self.mqtt_client_id = mqtt_client_id
        self.mqtt_debug = bool(mqtt_debug)
        self.mqtt_insecure = bool(mqtt_insecure)

        self._lock = threading.RLock()
        self._stop = threading.Event()

        # outputs for other classes
        self.latest_jpeg: Optional[bytes] = None
        self.mini_map = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        self.current_sector_states = ["unknown"] * self.sector_n
        self.current_decision = "STOP"
        self.reason = "init"

        # sensorhub (store in uart_* fields to keep your web UI unchanged)
        self.uart_dist_raw_cm: Optional[float] = None
        self.uart_dist_cm: Optional[float] = None
        self.uart_temp_c: Optional[float] = None
        self.uart_humid: Optional[float] = None

        # mqtt health
        self.mqtt_ok = False
        self.mqtt_error: Optional[str] = None
        self.mqtt_last_rx_ts: Optional[float] = None
        self.mqtt_dt_ms: Optional[int] = None
        self._mqtt_prev_rx: Optional[float] = None

        # flags
        self.imu_bump = False
        self.cam_blocked = False

        # ===== NEW camera tuning (stable) =====
        # ROI trapezoid (focus on floor / lower half)
        self.ROI_Y_TOP = 0.52          # start ROI a bit lower (camera mounted high)
        self.ROI_Y_BOT = 0.98
        self.ROI_TOP_RATIO = 0.50     # top width ratio
        self.ROI_BOT_RATIO = 1.00     # bottom width ratio

        # edge detection
        self.CANNY1, self.CANNY2 = 60, 160
        self.BLUR_K = (5, 5)

        # sector decision by edge density
        # (higher => more “blocked”)
        self.SECTOR_EDGE_TH = 0.060   # 0.04~0.08 thường hợp lý
        self.CENTER_BOOST = 1.10      # center stricter a bit

        # cam blocked heuristic
        self.BLUR_LAPLACE_TH = 30.0
        self.EDGE_DENSITY_MIN = 0.006

        self._threads: List[threading.Thread] = []

    def start(self):
        self._stop.clear()
        self._threads = []
        if self.enable_mqtt:
            self._threads.append(threading.Thread(target=self._mqtt_loop, daemon=True))
        if self.enable_camera:
            self._threads.append(threading.Thread(target=self._camera_loop, daemon=True))
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()

    def get_state(self) -> PerceptionState:
        with self._lock:
            return PerceptionState(
                ts=time.time(),
                sector_states=list(self.current_sector_states),
                decision=self.current_decision,
                reason=self.reason,
                uart_dist_raw_cm=self.uart_dist_raw_cm,
                uart_dist_cm=self.uart_dist_cm,
                uart_temp_c=self.uart_temp_c,
                uart_humid=self.uart_humid,
                mqtt_ok=bool(self.mqtt_ok),
                mqtt_error=self.mqtt_error,
                mqtt_last_rx_ts=self.mqtt_last_rx_ts,
                mqtt_dt_ms=self.mqtt_dt_ms,
                imu_bump=bool(self.imu_bump),
                cam_blocked=bool(self.cam_blocked),
                mode="AUTO",
                manual_override=None,
            )

    def get_status_dict(self) -> Dict:
        return _json_safe(asdict(self.get_state()))

    def get_mini_map_png(self) -> bytes:
        with self._lock:
            vis = cv2.resize(self.mini_map, (self.map_w * 10, self.map_h * 10), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", vis)
        return buf.tobytes() if ok else b""

    # ---------------- decision combine ----------------

    def compute_decision(self) -> Tuple[str, str]:
        with self._lock:
            dist = self.uart_dist_cm
            sector_states = list(self.current_sector_states)
            cam_blocked = bool(self.cam_blocked)

        # emergency stop always wins
        if dist is not None and dist < self.emergency_stop_cm:
            return "STOP", f"EMERGENCY_STOP({dist:.1f}cm)"

        # if camera is blocked/blurred, rely on distance only
        if cam_blocked:
            if dist is None:
                return "STOP", "CAM_BLOCKED_NO_DIST"
            if dist < self.safe_dist_cm:
                return "BACK", f"CAM_BLOCKED_NEAR({dist:.1f}cm)"
            return "FORWARD", "CAM_BLOCKED_DIST_OK"

        # near: choose turn by sectors
        if dist is not None and dist < self.safe_dist_cm:
            third = max(1, self.sector_n // 3)
            left = sector_states[:third]
            right = sector_states[-third:]
            left_free = sum(1 for s in left if s == "free")
            right_free = sum(1 for s in right if s == "free")

            if left_free > right_free and left_free > 0:
                return "TURN_LEFT", f"NEAR({dist:.1f})_LEFT_MORE_FREE"
            if right_free > left_free and right_free > 0:
                return "TURN_RIGHT", f"NEAR({dist:.1f})_RIGHT_MORE_FREE"

            # if both bad
            return "BACK", f"NEAR({dist:.1f})_NO_CLEAR"

        # normal camera navigation
        c = self.sector_n // 2
        center = sector_states[c] if sector_states else "unknown"
        if center == "free":
            return "FORWARD", "CENTER_FREE"

        left_free = sum(1 for s in sector_states[:c] if s == "free")
        right_free = sum(1 for s in sector_states[c + 1:] if s == "free")
        if right_free > left_free:
            return "TURN_RIGHT", "CENTER_BLOCKED_RIGHT_MORE_FREE"
        if left_free > right_free:
            return "TURN_LEFT", "CENTER_BLOCKED_LEFT_MORE_FREE"
        return "BACK", "CENTER_BLOCKED_NO_CLEAR"

    # ---------------- camera ----------------

    def _build_trapezoid_mask(self) -> np.ndarray:
        H, W = self.h, self.w
        y_top = int(H * self.ROI_Y_TOP)
        y_bot = int(H * self.ROI_Y_BOT)
        top_w = int(W * self.ROI_TOP_RATIO)
        bot_w = int(W * self.ROI_BOT_RATIO)
        xc = W // 2
        pts = np.array([
            [xc - top_w // 2, y_top],
            [xc + top_w // 2, y_top],
            [xc + bot_w // 2, y_bot],
            [xc - bot_w // 2, y_bot],
        ], dtype=np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _update_mini_map(self, sector_states: List[str]):
        self.mini_map[:-1, :, :] = self.mini_map[1:, :, :]
        row = np.zeros((self.map_w, 3), dtype=np.uint8)
        for i, s in enumerate(sector_states):
            if s == "blocked":
                row[i] = (0, 0, 255)
            elif s == "free":
                row[i] = (0, 180, 0)
            else:
                row[i] = (50, 50, 50)
        self.mini_map[-1, :, :] = row

    def _camera_loop(self):
        cap = cv2.VideoCapture(self.cam_dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[CAM] Cannot open camera: {self.cam_dev}")
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        roi_mask = self._build_trapezoid_mask()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            H, W = frame.shape[:2]

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray_full, cv2.CV_64F).var()

            gray = cv2.GaussianBlur(gray_full, self.BLUR_K, 0)
            edges = cv2.Canny(gray, self.CANNY1, self.CANNY2)

            # only ROI
            edges_roi = cv2.bitwise_and(edges, edges, mask=roi_mask)
            edge_density_all = float(np.mean(edges_roi > 0))
            cam_blocked = (lap_var < self.BLUR_LAPLACE_TH) or (edge_density_all < self.EDGE_DENSITY_MIN)

            # sector edge density
            sector_states = ["free"] * self.sector_n
            y0 = int(H * self.ROI_Y_TOP)
            y1 = int(H * self.ROI_Y_BOT)

            for i in range(self.sector_n):
                x0 = int(i * W / self.sector_n)
                x1 = int((i + 1) * W / self.sector_n)

                patch = edges_roi[y0:y1, x0:x1]
                if patch.size == 0:
                    sector_states[i] = "unknown"
                    continue

                den = float(np.mean(patch > 0))
                th = self.SECTOR_EDGE_TH
                if i == self.sector_n // 2:
                    th *= self.CENTER_BOOST

                sector_states[i] = "blocked" if den >= th else "free"

                # draw sector bar
                bar_y0 = int(H * 0.05)
                bar_y1 = int(H * 0.12)
                color = (0, 200, 0) if sector_states[i] == "free" else (0, 0, 255)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, bar_y0), (x1, bar_y1), color, -1)
                frame[bar_y0:bar_y1, x0:x1] = cv2.addWeighted(
                    overlay[bar_y0:bar_y1, x0:x1], 0.35,
                    frame[bar_y0:bar_y1, x0:x1], 0.65, 0
                )

            # draw ROI trapezoid outline
            # (optional visual)
            # cv2.polylines(frame, [np.column_stack(np.where(roi_mask > 0))], ... )  # too heavy
            # instead draw simple rectangle-ish guide:
            cv2.putText(frame, f"lapVar={lap_var:.1f} edgeDen={edge_density_all:.4f}", (10, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            with self._lock:
                self.cam_blocked = cam_blocked
                self.current_sector_states = sector_states
                self._update_mini_map(sector_states)

                decision, reason = self.compute_decision()
                self.current_decision = decision
                self.reason = reason

                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok2:
                    self.latest_jpeg = buf.tobytes()

            time.sleep(0.003)

        cap.release()

    # ---------------- MQTT ----------------

    def _parse_sensor_payload(self, payload: str) -> Optional[Tuple[float, float, float]]:
        payload = payload.strip()
        if not payload:
            return None

        # JSON: {"temp":29.6,"humid":20.0,"dist":11.4}
        if payload.startswith("{") and payload.endswith("}"):
            try:
                o = json.loads(payload)
                temp = float(o.get("temp", o.get("temperature", o.get("t"))))
                humid = float(o.get("humid", o.get("humidity", o.get("h"))))
                dist = float(o.get("dist", o.get("distance", o.get("d"))))
                return temp, humid, dist
            except Exception:
                return None

        # CSV: ts,temp,humid,dist
        if "," in payload:
            parts = [p.strip() for p in payload.split(",")]
            if len(parts) >= 4:
                try:
                    temp = float(parts[1])
                    humid = float(parts[2])
                    dist = float(parts[3])
                    return temp, humid, dist
                except Exception:
                    return None

        return None

    def _mqtt_loop(self):
        if mqtt is None:
            with self._lock:
                self.mqtt_ok = False
                self.mqtt_error = "paho-mqtt not installed"
            return

        def on_connect(client, userdata, flags, rc):
            with self._lock:
                self.mqtt_ok = (rc == 0)
                self.mqtt_error = None if rc == 0 else f"connect rc={rc}"
            if rc == 0:
                try:
                    client.subscribe(self.mqtt_topic, qos=0)
                    if self.mqtt_debug:
                        print(f"[MQTT] subscribed {self.mqtt_topic}")
                except Exception as e:
                    with self._lock:
                        self.mqtt_error = f"subscribe fail: {e}"

        def on_message(client, userdata, msg):
            try:
                payload = msg.payload.decode("utf-8", errors="ignore")
            except Exception:
                return

            parsed = self._parse_sensor_payload(payload)
            now = time.time()

            with self._lock:
                self.mqtt_last_rx_ts = now
                if self._mqtt_prev_rx is not None:
                    self.mqtt_dt_ms = int((now - self._mqtt_prev_rx) * 1000.0)
                self._mqtt_prev_rx = now
                self.mqtt_ok = True
                self.mqtt_error = None

            if parsed:
                temp, humid, dist = parsed
                with self._lock:
                    self.uart_temp_c = temp
                    self.uart_humid = humid
                    self.uart_dist_raw_cm = dist
                    self.uart_dist_cm = dist  # simple (no filter)
            else:
                if self.mqtt_debug:
                    print("[MQTT] unparsed payload:", payload[:120])

        def on_disconnect(client, userdata, rc):
            with self._lock:
                self.mqtt_ok = False
                self.mqtt_error = f"disconnected rc={rc}"

        client = mqtt.Client(client_id=self.mqtt_client_id, clean_session=True)
        if self.mqtt_user:
            client.username_pw_set(self.mqtt_user, self.mqtt_pass)

        # TLS insecure mode (no cert verify) because you asked
        try:
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True if self.mqtt_insecure else False)
        except Exception as e:
            with self._lock:
                self.mqtt_error = f"tls setup fail: {e}"

        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect

        while not self._stop.is_set():
            try:
                if self.mqtt_debug:
                    print(f"[MQTT] connecting {self.mqtt_host}:{self.mqtt_port} ...")
                client.connect(self.mqtt_host, self.mqtt_port, keepalive=30)
                client.loop_start()

                # keep thread alive
                while not self._stop.is_set():
                    time.sleep(0.2)

                break
            except Exception as e:
                with self._lock:
                    self.mqtt_ok = False
                    self.mqtt_error = f"connect fail: {e}"
                time.sleep(2.0)

        try:
            client.loop_stop()
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass
