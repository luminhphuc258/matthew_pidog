#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

try:
    import serial
except Exception:
    serial = None

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
    uart_dist_cm: Optional[float] = None
    uart_temp_c: Optional[float] = None
    uart_humid: Optional[float] = None
    imu_bump: bool = False
    cam_blocked: bool = False


class PerceptionPlanner:
    """
    - Camera obstacle detection -> sector_states (9 sectors)
    - Mini-map 2D time history (robot at center logic: dùng history dạng "rows")
    - UART ESP32 (timestamp,temp,humid,ultrasonic_cm)
    - IMU bump (optional)
    -> compute decision
    """

    def __init__(
        self,
        cam_dev="/dev/video0",
        w=640,
        h=480,
        fps=30,
        sector_n=9,
        map_h=80,
        serial_port="/dev/ttyUSB0",
        baud=115200,
        safe_dist_cm=50.0,
        enable_imu=False,
        i2c_bus=1,
        addr_acc=0x36,
    ):
        self.cam_dev, self.w, self.h, self.fps = cam_dev, w, h, fps
        self.sector_n = sector_n
        self.map_h = map_h
        self.map_w = sector_n

        self.serial_port, self.baud = serial_port, baud
        self.safe_dist_cm = float(safe_dist_cm)

        self.enable_imu = enable_imu
        self.i2c_bus = i2c_bus
        self.addr_acc = addr_acc

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # outputs for other classes
        self.latest_jpeg: Optional[bytes] = None
        self.mini_map = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        self.current_sector_states = ["unknown"] * self.sector_n
        self.current_decision = "STOP"
        self.reason = "init"

        # UART readings
        self.uart_dist_cm: Optional[float] = None
        self.uart_temp_c: Optional[float] = None
        self.uart_humid: Optional[float] = None

        self.imu_bump = False
        self.cam_blocked = False

        # tuning
        self.CANNY1, self.CANNY2 = 50, 150
        self.DILATE_ITER = 2
        self.BLUR_K = (5, 5)

        self.TRAP_Y_TOP = 0.45
        self.TRAP_TOP_RATIO = 0.55
        self.TRAP_BOTTOM_RATIO = 1.00

        self.MIN_AREA = 300
        self.MIN_H_RATIO = 0.18
        self.MIN_ASPECT = 1.6
        self.NEAR_BOTTOM = 0.80

        self.NEAR_AREA_RATIO = 0.060
        self.NEAR_H_RATIO = 0.28
        self.NEAR_BOTTOM_STRICT = 0.88

        self.BLUR_TH = 35.0
        self.EDGE_LOW_TH = 0.010
        self.BLOCK_BOX_TH = 0.45

        self._threads: List[threading.Thread] = []

    def start(self):
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._camera_loop, daemon=True),
            threading.Thread(target=self._uart_loop, daemon=True),
        ]
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
                uart_dist_cm=self.uart_dist_cm,
                uart_temp_c=self.uart_temp_c,
                uart_humid=self.uart_humid,
                imu_bump=self.imu_bump,
                cam_blocked=self.cam_blocked,
            )

    def get_status_dict(self) -> Dict:
        st = self.get_state()
        return asdict(st)

    def get_mini_map_png(self) -> bytes:
        with self._lock:
            vis = cv2.resize(self.mini_map, (self.map_w * 10, self.map_h * 10), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", vis)
        return buf.tobytes() if ok else b""

    # ---------------- internal ----------------

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

    def _compute_decision(self, sector_states: List[str]) -> Tuple[str, str]:
        if self.imu_bump:
            return "BACK", "IMU_BUMP"
        if self.cam_blocked:
            return "BACK", "CAM_BLOCKED"

        if self.uart_dist_cm is not None and self.uart_dist_cm < self.safe_dist_cm:
            left = sector_states[: self.sector_n // 3]
            center = sector_states[self.sector_n // 3: 2 * self.sector_n // 3]
            right = sector_states[2 * self.sector_n // 3:]

            if all(s != "blocked" for s in left):
                return "TURN_LEFT", f"UART_NEAR({self.uart_dist_cm:.1f}cm)"
            if all(s != "blocked" for s in right):
                return "TURN_RIGHT", f"UART_NEAR({self.uart_dist_cm:.1f}cm)"
            return "STOP", f"UART_NEAR_BLOCKED({self.uart_dist_cm:.1f}cm)"

        center_idx = self.sector_n // 2
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

    def _camera_loop(self):
        cap = cv2.VideoCapture(self.cam_dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.cam_dev}")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        floor_mask = self._build_trapezoid_mask()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray_full, cv2.CV_64F).var()
            edges_full = cv2.Canny(gray_full, self.CANNY1, self.CANNY2)
            edge_density = float(np.mean(edges_full > 0))
            self.cam_blocked = (lap_var < self.BLUR_TH) or (edge_density < self.EDGE_LOW_TH)

            gray = cv2.GaussianBlur(gray_full, self.BLUR_K, 0)
            edges = cv2.Canny(gray, self.CANNY1, self.CANNY2)
            edges = cv2.dilate(edges, kernel, iterations=self.DILATE_ITER)
            edges = cv2.bitwise_and(edges, edges, mask=floor_mask)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            sector_states = ["free"] * self.sector_n
            H, W = frame.shape[:2]

            bboxes = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.MIN_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                h_ratio = h / float(H)
                if h_ratio < self.MIN_H_RATIO:
                    continue
                aspect = (h / float(w + 1e-6))
                if aspect < self.MIN_ASPECT:
                    continue
                if (y + h) / float(H) < self.NEAR_BOTTOM:
                    continue
                bboxes.append((x, y, w, h, area))

            for (x, y, ww, hh, area) in bboxes:
                x_center = x + ww / 2.0
                sec = int(x_center / W * self.sector_n)
                sec = max(0, min(self.sector_n - 1, sec))
                sector_states[sec] = "blocked"

                if (area / float(W * H) > self.NEAR_AREA_RATIO) or (hh / float(H) > self.NEAR_H_RATIO):
                    for nb in (sec - 1, sec + 1):
                        if 0 <= nb < self.sector_n:
                            sector_states[nb] = "blocked"

                cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 0, 255), 2)

            y0 = int(H * 0.05)
            y1 = int(H * 0.12)
            for i, s in enumerate(sector_states):
                x0 = int(i * W / self.sector_n)
                x1 = int((i + 1) * W / self.sector_n)
                color = (0, 200, 0) if s == "free" else (0, 0, 255)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
                frame[y0:y1, x0:x1] = cv2.addWeighted(
                    overlay[y0:y1, x0:x1], 0.35, frame[y0:y1, x0:x1], 0.65, 0
                )

            with self._lock:
                self.current_sector_states = sector_states
                self._update_mini_map(sector_states)

                decision, reason = self._compute_decision(sector_states)
                self.current_decision, self.reason = decision, reason

                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok2:
                    self.latest_jpeg = buf.tobytes()

            time.sleep(0.001)

        cap.release()

    def _uart_loop(self):
        if serial is None:
            return
        try:
            ser = serial.Serial(self.serial_port, self.baud, timeout=1)
            time.sleep(1.5)
        except Exception:
            return

        while not self._stop.is_set():
            try:
                raw = ser.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # format: timestamp,temp,humidity,ultrasonic_cm
                parts = line.split(",")
                if len(parts) >= 4:
                    temp_c = float(parts[1])
                    humid = float(parts[2])
                    dist = float(parts[3])

                    with self._lock:
                        self.uart_temp_c = temp_c
                        self.uart_humid = humid
                        self.uart_dist_cm = dist

            except Exception:
                time.sleep(0.05)

        try:
            ser.close()
        except Exception:
            pass

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
                bump = False  # TODO: replace by real SH3001 bump logic
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
