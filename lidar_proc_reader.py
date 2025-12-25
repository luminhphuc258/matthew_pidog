import json
import math
import threading
import subprocess
import time
from typing import List, Tuple, Optional


class LidarProcReader:
    """
    Spawn ./lidar_bridge and read JSON scan frames from stdout.
    """
    def __init__(self, bridge_path="./lidar_bridge", port="/dev/ttyUSB0", baud=460800, max_points=2500):
        self.bridge_path = bridge_path
        self.port = port
        self.baud = int(baud)
        self.max_points = int(max_points)

        self._proc: Optional[subprocess.Popen] = None
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._points_xy: List[Tuple[float, float]] = []
        self._last_ts = 0.0
        self._last_err: Optional[str] = None

    def start(self):
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._t:
            self._t.join(timeout=2)

    def get_points(self):
        with self._lock:
            return list(self._points_xy), self._last_ts, self._last_err

    def _run(self):
        while not self._stop.is_set():
            try:
                self._last_err = None

                self._proc = subprocess.Popen(
                    [self.bridge_path, self.port, str(self.baud)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

                assert self._proc.stdout is not None

                for line in self._proc.stdout:
                    if self._stop.is_set():
                        break
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)
                    pts = data.get("pts", [])

                    # convert to xy (meters)
                    xy = []
                    for a_deg, dist_m, q in pts:
                        if dist_m <= 0:
                            continue
                        a = math.radians(float(a_deg))
                        r = float(dist_m)
                        x = r * math.cos(a)
                        y = r * math.sin(a)
                        xy.append((x, y))

                    if len(xy) > self.max_points:
                        step = max(1, len(xy) // self.max_points)
                        xy = xy[::step]

                    with self._lock:
                        self._points_xy = xy
                        self._last_ts = time.time()

                # if exited
                rc = self._proc.poll()
                self._last_err = f"bridge exited rc={rc}"
                time.sleep(1)

            except Exception as e:
                self._last_err = str(e)
                time.sleep(1)
