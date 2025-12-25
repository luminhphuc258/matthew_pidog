#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import threading
from typing import List, Tuple, Optional

import numpy as np

try:
    from rplidar import RPLidar  # pip install rplidar
except Exception as e:
    raise SystemExit("Missing package rplidar. Run: pip3 install rplidar") from e


class LidarReader:
    """
    Continuously read scans from RPLidar and keep the latest points (x,y) in meters.
    """

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, max_points=2000):
        self.port = port
        self.baudrate = baudrate
        self.max_points = max_points

        self._lidar: Optional[RPLidar] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._points_xy_m: List[Tuple[float, float]] = []
        self._last_ts = 0.0
        self._last_err: Optional[str] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

        if self._lidar:
            try:
                self._lidar.stop()
            except Exception:
                pass
            try:
                self._lidar.stop_motor()
            except Exception:
                pass
            try:
                self._lidar.disconnect()
            except Exception:
                pass

    def get_points(self) -> Tuple[List[Tuple[float, float]], float, Optional[str]]:
        with self._lock:
            return list(self._points_xy_m), self._last_ts, self._last_err

    def _connect(self):
        self._lidar = RPLidar(self.port, baudrate=self.baudrate, timeout=1)
        # Some models need motor start; rplidar lib handles it, but we force:
        try:
            self._lidar.start_motor()
        except Exception:
            pass

        # quick info (optional)
        try:
            info = self._lidar.get_info()
            print("LIDAR info:", info)
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                self._last_err = None
                if self._lidar is None:
                    self._connect()

                # iter_scans yields list of (quality, angle, distance_mm)
                for scan in self._lidar.iter_scans(max_buf_meas=2000):
                    if self._stop.is_set():
                        break

                    pts = []
                    for quality, angle_deg, dist_mm in scan:
                        if dist_mm <= 0:
                            continue
                        # filter too close / too far (tune as you like)
                        if dist_mm < 150 or dist_mm > 12000:
                            continue

                        a = math.radians(angle_deg)
                        r = dist_mm / 1000.0  # meters
                        x = r * math.cos(a)
                        y = r * math.sin(a)
                        pts.append((x, y))

                    # downsample + cap
                    if len(pts) > self.max_points:
                        step = max(1, len(pts) // self.max_points)
                        pts = pts[::step]

                    with self._lock:
                        self._points_xy_m = pts
                        self._last_ts = time.time()

            except Exception as e:
                self._last_err = str(e)
                print("LidarReader error:", self._last_err)
                # try reconnect
                try:
                    if self._lidar:
                        try:
                            self._lidar.stop()
                        except Exception:
                            pass
                        try:
                            self._lidar.stop_motor()
                        except Exception:
                            pass
                        try:
                            self._lidar.disconnect()
                        except Exception:
                            pass
                except Exception:
                    pass
                self._lidar = None
                time.sleep(1.0)
