#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

try:
    from smbus2 import SMBus
except Exception:
    SMBus = None


@dataclass
class PerceptionState:
    ts: float
    sector_states: List[str]          # len=9: "free" | "blocked" | "unknown"
    decision: str                    # "FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "BACK" | "STOP"
    reason: str

    # SensorHub data (giữ tên uart_* để main/web cũ vẫn chạy)
    uart_dist_raw_cm: Optional[float] = None
    uart_dist_cm: Optional[float] = None
    uart_temp_c: Optional[float] = None
    uart_humid: Optional[float] = None

    # MQTT health
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


class PerceptionPlanner:
    """
    - Camera obstacle detection -> N sectors
    - SensorHub via MQTT -> topic /pidog/sensorhubdata
      payload JSON (example):
        {"ts_ms":123,"temp_c":29.6,"humid":20.0,"dist_cm":11.44}
    - Priority: DIST first, then camera
    """

    def __init__(
        self,
        # camera
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        enable_camera=True,

        # decision thresholds
        safe_dist_cm=50.0,
        emergency_stop_cm=10.0,

        # mqtt
        enable_mqtt=True,
        mqtt_host="rfff7184.ala.us-east-1.emqxsl.com",
        mqtt_port=8883,
        mqtt_user="robot_matthew",
        mqtt_pass="29061992abCD!yesokmen",
        mqtt_topic="/pidog/sensorhubdata",
        mqtt_client_id="pidog-perception-pi",
        mqtt_debug=False,

        # imu
        enable_imu=False,
        i2c_bus=1,
        addr_acc=0x36,
    ):
        # camera
        self.cam_dev, self.w, self.h, self.fps = cam_dev, w, h, fps
        self.sector_n = int(sector_n)
        self.map_h = int(map_h)
        self.map_w = self.sector_n
        self.enable_camera = bool(enable_camera)

        # thresholds
        self.safe_dist_cm = float(safe_dist_cm)
        self.emergency_stop_cm = float(emergency_stop_cm)

        # mqtt
        self.enable_mqtt = bool(enable_mqtt)
        self.mqtt_host = mqtt_host
        self.mqtt_port = int(mqtt_port)
        self.mqtt_user = mqtt_user
        self.mqtt_pass = mqtt_pass
        self.mqtt_topic = mqtt_topic
        self.mqtt_client_id = mqtt_client_id
        self.mqtt_debug = bool(mqtt_debug)

        # imu
        self.enable_imu = bool(enable_imu)
        self.i2c_bus = i2c_bus
        self.addr_acc = addr_acc

        self._lock = threading.RLock()
        self._stop = threading.Event()

        # outputs for other classes
        self.latest_jpeg: Optional[bytes] = None
        self.mini_map = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        self.current_sector_states = ["unknown"] * self.sector_n
        self.current_decision = "STOP"
        self.reason = "init"

        # sensorhub values (mapped to uart_* for compatibility)
        self.uart_dist_raw_cm: Optional[float] = None
        self.uart_dist_cm: Optional[float] = None
        self.uart_temp_c: Optional[float] = None
        self.uart_humid: Optional[float] = None

        # mqtt health
        self.mqtt_ok: bool = False
        self.mqtt_error: Optional[str] = None
        self.mqtt_last_rx_ts: Optional[float] = None
        self.mqtt_dt_ms: Optional[int] = None
        self._mqtt_prev_rx_ts: Optional[float] = None

        # flags
        self.imu_bump = False
        self.cam_blocked = False

        # camera tuning (bạn có thể chỉnh dần)
        self.CANNY1, self.CANNY2 = 40, 120
        self.DILATE_ITER = 2
        self.BLUR_K = (5, 5)

        # ROI trapezoid: nâng vùng nhìn lên chút để bắt vật cao (kệ/ bàn)
        self.TRAP_Y_TOP = 0.25          # (cũ 0.45) -> nhìn cao hơn
        self.TRAP_TOP_RATIO = 0.70      # (cũ 0.55)
        self.TRAP_BOTTOM_RATIO = 1.00

        # contour filter: bớt “khắt khe” để bắt vật như bàn/kệ
        self.MIN_AREA = 260
        self.MIN_H_RATIO = 0.10         # (cũ 0.18)
        self.NEAR_BOTTOM = 0.60         # (cũ 0.80) -> không bắt buộc sát đáy

        # blur/edge detect camera blocked
        self.BLUR_TH = 30.0
        self.EDGE_LOW_TH = 0.008

        self._threads: List[threading.Thread] = []

    # ---------------- public ----------------

    def start(self):
        self._stop.clear()
        self._threads = []
        if self.enable_mqtt:
            self._threads.append(threading.Thread(target=self._mqtt_loop, daemon=True))
        if self.enable_camera:
            self._threads.append(threading.Thread(target=self._camera_loop, daemon=True))
        if self.enable_imu:
            self._threads.append(threading.Thread(target=self._imu_loop, daemon=True))

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
                mqtt_ok=self.mqtt_ok,
                mqtt_error=self.mqtt_error,
                mqtt_last_rx_ts=self.mqtt_last_rx_ts,
                mqtt_dt_ms=self.mqtt_dt_ms,
                imu_bump=self.imu_bump,
                cam_blocked=self.cam_blocked,
                mode="AUTO",
                manual_override=None,
            )

    def get_status_dict(self) -> Dict:
        return asdict(self.get_state())

    def get_mini_map_png(self) -> bytes:
        with self._lock:
            vis = cv2.resize(self.mini_map, (self.map_w * 10, self.map_h * 10), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", vis)
        return buf.tobytes() if ok else b""

    # ---------------- decision combine ----------------

    def compute_decision(self) -> Tuple[str, str]:
        """
        Priority:
        1) IMU bump
        2) DIST emergency stop
        3) DIST near avoidance (camera used only to pick turn direction)
        4) camera normal navigation
        """
        with self._lock:
            dist = self.uart_dist_cm
            sector_states = list(self.current_sector_states)
            cam_blocked = self.cam_blocked
            imu_bump = self.imu_bump

        if imu_bump:
            return "BACK", "IMU_BUMP"

        # 1) emergency stop always wins (dist ưu tiên cao nhất)
        if dist is not None and dist < self.emergency_stop_cm:
            return "STOP", f"EMERGENCY_STOP({dist:.2f}cm)"

        # 2) if distance says near -> decide turn (use camera only for direction)
        if dist is not None and dist < self.safe_dist_cm:
            # nếu camera bị block / chưa có sector -> ưu tiên TURN_RIGHT để thoát
            if (not sector_states) or all(s == "unknown" for s in sector_states) or cam_blocked:
                return "TURN_RIGHT", f"DIST_NEAR({dist:.2f}cm)_CAM_UNRELIABLE"

            n = len(sector_states)
            c = n // 2
            left = sector_states[:c]
            right = sector_states[c+1:]
            left_free = sum(1 for s in left if s == "free")
            right_free = sum(1 for s in right if s == "free")

            if left_free > right_free and left_free > 0:
                return "TURN_LEFT", f"DIST_NEAR({dist:.2f}cm)_TURN_LEFT"
            if right_free > left_free and right_free > 0:
                return "TURN_RIGHT", f"DIST_NEAR({dist:.2f}cm)_TURN_RIGHT"

            # không rõ -> lùi nhẹ
            return "BACK", f"DIST_NEAR({dist:.2f}cm)_NO_CLEAR"

        # 3) camera blocked (nhưng dist không near) -> cứ đứng quan sát/đi chậm
        if cam_blocked:
            return "STOP", "CAM_BLOCKED"

        # 4) normal camera rule
        if not sector_states:
            return "STOP", "NO_SECTORS"

        center_idx = len(sector_states) // 2
        center = sector_states[center_idx]
        if center == "free":
            return "FORWARD", "CENTER_FREE"

        left_free = sum(1 for s in sector_states[:center_idx] if s == "free")
        right_free = sum(1 for s in sector_states[center_idx + 1:] if s == "free")
        if left_free > right_free:
            return "TURN_LEFT", "CENTER_BLOCKED_LEFT_MORE_FREE"
        if right_free > left_free:
            return "TURN_RIGHT", "CENTER_BLOCKED_RIGHT_MORE_FREE"
        return "BACK", "CENTER_BLOCKED_NO_CLEAR"

    # ---------------- camera ----------------

    def _build_trapezoid_mask(self) -> np.ndarray:
        H, W = self.h, self.w
        y_top = int(H * self.TRAP_Y_TOP)
        y_bot = int(H * self.TRAP_BOTTOM_RATIO)

        top_w = int(W * self.TRAP_TOP_RATIO)
        bot_w = int(W * 1.00)

        x_center = W // 2
        pts = np.array([
            [x_center - top_w // 2, y_top],
            [x_center + top_w // 2, y_top],
            [x_center + bot_w // 2, y_bot],
            [x_center - bot_w // 2, y_bot],
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

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        roi_mask = self._build_trapezoid_mask()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            H, W = frame.shape[:2]
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # camera health
            lap_var = cv2.Laplacian(gray_full, cv2.CV_64F).var()
            edges_full = cv2.Canny(gray_full, self.CANNY1, self.CANNY2)
            edge_density = float(np.mean(edges_full > 0))
            cam_blocked = (lap_var < self.BLUR_TH) or (edge_density < self.EDGE_LOW_TH)

            # detect edges in ROI
            gray = cv2.GaussianBlur(gray_full, self.BLUR_K, 0)
            edges = cv2.Canny(gray, self.CANNY1, self.CANNY2)
            edges = cv2.dilate(edges, kernel, iterations=self.DILATE_ITER)
            edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            sector_states = ["free"] * self.sector_n
            bboxes = []

            for c in contours:
                area = cv2.contourArea(c)
                if area < self.MIN_AREA:
                    continue
                x, y, ww, hh = cv2.boundingRect(c)
                h_ratio = hh / float(H)
                if h_ratio < self.MIN_H_RATIO:
                    continue
                if (y + hh) / float(H) < self.NEAR_BOTTOM:
                    continue
                bboxes.append((x, y, ww, hh, area))

            for (x, y, ww, hh, area) in bboxes:
                x_center = x + ww / 2.0
                sec = int(x_center / W * self.sector_n)
                sec = max(0, min(self.sector_n - 1, sec))
                sector_states[sec] = "blocked"

                # mark neighbors (thêm độ “dày” obstacle)
                for nb in (sec - 1, sec + 1):
                    if 0 <= nb < self.sector_n:
                        sector_states[nb] = "blocked"

                cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 0, 255), 2)

            # overlay sectors bar
            y0 = int(H * 0.05)
            y1 = int(H * 0.12)
            for i, s in enumerate(sector_states):
                x0 = int(i * W / self.sector_n)
                x1 = int((i + 1) * W / self.sector_n)
                color = (0, 200, 0) if s == "free" else (0, 0, 255)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
                frame[y0:y1, x0:x1] = cv2.addWeighted(
                    overlay[y0:y1, x0:x1], 0.35,
                    frame[y0:y1, x0:x1], 0.65, 0
                )

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

            time.sleep(0.002)

        cap.release()

    # ---------------- MQTT SensorHub ----------------

    def _mqtt_loop(self):
        if mqtt is None:
            with self._lock:
                self.mqtt_ok = False
                self.mqtt_error = "paho-mqtt not installed"
            return

        def on_connect(client, userdata, flags, rc):
            if self.mqtt_debug:
                print("[MQTT] on_connect rc =", rc)
            with self._lock:
                self.mqtt_ok = (rc == 0)
                self.mqtt_error = None if rc == 0 else f"connect rc={rc}"
            if rc == 0:
                try:
                    client.subscribe(self.mqtt_topic, qos=0)
                    if self.mqtt_debug:
                        print("[MQTT] subscribed:", self.mqtt_topic)
                except Exception as e:
                    with self._lock:
                        self.mqtt_error = f"subscribe fail: {e}"

        def on_message(client, userdata, msg):
            now = time.time()
            try:
                payload = msg.payload.decode("utf-8", errors="ignore").strip()
                # payload JSON: {"ts_ms":..,"temp_c":..,"humid":..,"dist_cm":..}
                import json
                j = json.loads(payload)

                temp = j.get("temp_c", None)
                humid = j.get("humid", None)
                dist = j.get("dist_cm", None)

                with self._lock:
                    if temp is not None:
                        self.uart_temp_c = float(temp)
                    if humid is not None:
                        self.uart_humid = float(humid)
                    if dist is not None:
                        self.uart_dist_raw_cm = float(dist)
                        self.uart_dist_cm = float(dist)

                    self.mqtt_ok = True
                    self.mqtt_error = None
                    self.mqtt_last_rx_ts = now

                    if self._mqtt_prev_rx_ts is not None:
                        self.mqtt_dt_ms = int((now - self._mqtt_prev_rx_ts) * 1000)
                    self._mqtt_prev_rx_ts = now

            except Exception as e:
                with self._lock:
                    self.mqtt_ok = False
                    self.mqtt_error = f"parse fail: {e}"

        def on_disconnect(client, userdata, rc):
            if self.mqtt_debug:
                print("[MQTT] disconnected rc =", rc)
            with self._lock:
                self.mqtt_ok = False
                if rc != 0:
                    self.mqtt_error = f"disconnect rc={rc}"

        client = mqtt.Client(client_id=self.mqtt_client_id, clean_session=True)
        client.username_pw_set(self.mqtt_user, self.mqtt_pass)

        # Port 8883 => TLS, nhưng cho phép insecure như ESP32 setInsecure()
        try:
            client.tls_set()  # use system CA (but we will allow insecure)
            client.tls_insecure_set(True)
        except Exception as e:
            with self._lock:
                self.mqtt_ok = False
                self.mqtt_error = f"tls init fail: {e}"
            return

        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect

        # auto reconnect loop
        while not self._stop.is_set():
            try:
                client.connect(self.mqtt_host, self.mqtt_port, keepalive=30)
                client.loop_start()

                # watchdog: nếu quá lâu không nhận data -> mqtt_ok = False
                while not self._stop.is_set():
                    time.sleep(0.5)
                    with self._lock:
                        if self.mqtt_last_rx_ts is not None:
                            if (time.time() - self.mqtt_last_rx_ts) > 3.0:
                                self.mqtt_ok = False
                                self.mqtt_error = "no data > 3s"

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

    # ---------------- IMU ----------------

    def _imu_loop(self):
        if SMBus is None:
            return
        try:
            bus = SMBus(self.i2c_bus)
        except Exception:
            return

        last_bump = 0.0
        while not self._stop.is_set():
            try:
                bump = False  # TODO: thay bằng SH3001 thật
                now = time.time()
                if bump:
                    last_bump = now
                with self._lock:
                    self.imu_bump = (now - last_bump) < 1.2
            except Exception:
                pass
            time.sleep(0.02)

        try:
            bus.close()
        except Exception:
            pass
