#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, threading, re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

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
    sector_states: List[str]
    decision: str
    reason: str
    uart_dist_cm: Optional[float] = None
    uart_temp_c: Optional[float] = None
    uart_humid: Optional[float] = None
    imu_bump: bool = False
    cam_blocked: bool = False


class PerceptionPlanner:
    """
    - UART from N8R8/ESP32 (text Vietnamese):
        "UART >> Nhiệt độ: 28.50°C | Độ ẩm: 20.00%"
        "UART >> Khoảng cách vật cản: 6.82 cm"
      -> store uart_temp_c / uart_humid / uart_dist_cm

    - Camera + mini-map: optional (enable_camera)
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
        enable_camera=True,          # ✅ cho phép tắt camera khi test UART
        uart_debug=False,            # ✅ in raw line UART
        uart_print_every=0.5,        # ✅ tần suất in raw
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

        self.enable_camera = bool(enable_camera)
        self.uart_debug = bool(uart_debug)
        self.uart_print_every = float(uart_print_every)

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # outputs
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

        # camera tuning (giữ placeholder; dùng khi enable_camera=True)
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
        self.BLUR_TH = 35.0
        self.EDGE_LOW_TH = 0.010

        self._threads: List[threading.Thread] = []

        # regex parse numbers
        # Dòng temp/hum: lấy số trước °C và số trước %
        self._re_temp = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*°?\s*C", re.IGNORECASE)
        self._re_hum = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
        # Dòng distance: lấy số trước "cm"
        self._re_cm = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)

    def start(self):
        self._stop.clear()
        self._threads = [threading.Thread(target=self._uart_loop, daemon=True)]

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
                uart_dist_cm=self.uart_dist_cm,
                uart_temp_c=self.uart_temp_c,
                uart_humid=self.uart_humid,
                imu_bump=self.imu_bump,
                cam_blocked=self.cam_blocked,
            )

    def get_status_dict(self) -> Dict:
        return asdict(self.get_state())

    # ---------------- UART ----------------

    def _parse_uart_line(self, line: str):
        """
        Parse Vietnamese text lines:
          - "UART >> Nhiệt độ: 28.50°C | Độ ẩm: 20.00%"
          - "UART >> Khoảng cách vật cản: 6.82 cm"
        Also works even if dấu tiếng Việt bị lỗi (chỉ cần có số + °C/%/cm).
        """
        if not line:
            return

        # TEMP + HUM
        # Nếu dòng có °C và % thì coi là temp/hum
        if ("c" in line.lower()) and ("%" in line):
            mt = self._re_temp.search(line)
            mh = self._re_hum.search(line)
            if mt:
                try:
                    temp_c = float(mt.group(1))
                    with self._lock:
                        self.uart_temp_c = temp_c
                except Exception:
                    pass
            if mh:
                try:
                    hum = float(mh.group(1))
                    with self._lock:
                        self.uart_humid = hum
                except Exception:
                    pass
            return

        # DISTANCE
        if "cm" in line.lower():
            md = self._re_cm.search(line)
            if md:
                try:
                    dist_cm = float(md.group(1))
                    with self._lock:
                        self.uart_dist_cm = dist_cm
                except Exception:
                    pass

    def _uart_loop(self):
        if serial is None:
            print("[UART] pyserial not installed. Run: pip3 install pyserial")
            return

        try:
            ser = serial.Serial(self.serial_port, self.baud, timeout=1)
            time.sleep(1.5)
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            print(f"[UART] opened {self.serial_port} @ {self.baud}")
        except Exception as e:
            print(f"[UART] cannot open {self.serial_port}: {e}")
            return

        last_dbg = 0.0
        while not self._stop.is_set():
            try:
                raw = ser.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if self.uart_debug:
                    now = time.time()
                    if (now - last_dbg) >= self.uart_print_every:
                        print("UART >>", line)
                        last_dbg = now

                self._parse_uart_line(line)

                # decision tối giản khi chỉ test UART
                with self._lock:
                    # Nếu có distance và quá gần thì STOP, còn lại FORWARD (tuỳ bạn)
                    if self.uart_dist_cm is not None and self.uart_dist_cm < self.safe_dist_cm:
                        self.current_decision = "STOP"
                        self.reason = f"UART_NEAR({self.uart_dist_cm:.2f}cm)"
                    else:
                        self.current_decision = "FORWARD"
                        self.reason = "UART_OK"

            except Exception:
                time.sleep(0.05)

        try:
            ser.close()
        except Exception:
            pass

    # ---------------- Camera (optional) ----------------

    def _camera_loop(self):
        if cv2 is None:
            print("[CAM] opencv not available, camera loop disabled")
            return

        cap = cv2.VideoCapture(self.cam_dev)
        if not cap.isOpened():
            print(f"[CAM] Cannot open camera: {self.cam_dev}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # Placeholder: bạn có thể gắn lại logic sector/canny như bản cũ sau
            with self._lock:
                self.current_sector_states = ["unknown"] * self.sector_n
                self.cam_blocked = False

            time.sleep(0.02)

        cap.release()

    # ---------------- IMU (optional placeholder) ----------------

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
                bump = False  # TODO: replace with real logic
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
