#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import statistics as stats
import threading
import time
from typing import Optional, Dict, Any, Tuple, List

from smbus2 import SMBus
from flask import Flask, jsonify, Response, request

# =========================
# CONFIG
# =========================
BUS_ID = 1
ADDR_ACC = 0x36

REG_AX = 0x01
REG_AY = 0x03
REG_AZ = 0x05
REG_GX = 0x07
REG_GY = 0x09
REG_GZ = 0x0B

DT = 0.02          # 50 Hz
CALIB_S = 3.0
COOLDOWN_S = 0.8

# sensitivity (tune)
K_dA = 8.0
K_jerk = 8.0
K_gyro = 8.0

MIN_dA_TH = 30.0
MIN_JERK_TH = 1500.0
MIN_GYRO_TH = 10.0

REQUIRE_TWO_SIGNALS = True

# output label behavior
YES_HOLD_S = 0.35          # giữ "yes" trong 0.35s sau khi phát hiện va chạm
ERROR_NO_FORCE = True      # nếu lỗi sensor => label "no"
ERROR_LOG_COOLDOWN = 2.0   # tránh spam log lỗi

# HTTP server
HOST = "127.0.0.1"
PORT = 9411

# =========================
# Collision Detector (based on your code)
# =========================
class CollisionDetector:
    def __init__(
        self,
        bus_id: int = BUS_ID,
        addr: int = ADDR_ACC,
        dt: float = DT,
        calib_s: float = CALIB_S,
        cooldown_s: float = COOLDOWN_S,
        k_dA: float = K_dA,
        k_jerk: float = K_jerk,
        k_gyro: float = K_gyro,
        min_dA_th: float = MIN_dA_TH,
        min_jerk_th: float = MIN_JERK_TH,
        min_gyro_th: float = MIN_GYRO_TH,
        require_two_signals: bool = REQUIRE_TWO_SIGNALS,
        name: str = "SH3001",
    ):
        self.addr = addr
        self.DT = dt
        self.CALIB_S = calib_s
        self.COOLDOWN_S = cooldown_s

        self.K_dA = k_dA
        self.K_jerk = k_jerk
        self.K_gyro = k_gyro

        self.MIN_dA_TH = min_dA_th
        self.MIN_JERK_TH = min_jerk_th
        self.MIN_GYRO_TH = min_gyro_th

        self.require_two_signals = require_two_signals
        self.name = name

        self.bus = SMBus(bus_id)

        self.bias_gx = 0.0
        self.bias_gy = 0.0
        self.bias_gz = 0.0

        self.dA_th = self.MIN_dA_TH
        self.jerk_th = self.MIN_JERK_TH
        self.gyro_th = self.MIN_GYRO_TH

        self.last_ax = None
        self.last_ay = None
        self.last_az = None

        self.last_event_t = 0.0
        self.ready = False

    def _read_word(self, reg: int) -> int:
        hi = self.bus.read_byte_data(self.addr, reg)
        lo = self.bus.read_byte_data(self.addr, reg + 1)
        val = (hi << 8) | lo
        if val & 0x8000:
            val -= 65536
        return val

    def read_accel_raw(self) -> Tuple[int, int, int]:
        ax = self._read_word(REG_AX)
        ay = self._read_word(REG_AY)
        az = self._read_word(REG_AZ)
        return ax, ay, az

    def read_gyro_raw(self) -> Tuple[int, int, int]:
        gx = self._read_word(REG_GX)
        gy = self._read_word(REG_GY)
        gz = self._read_word(REG_GZ)
        return gx, gy, gz

    @staticmethod
    def mag3(x: float, y: float, z: float) -> float:
        return math.sqrt(x * x + y * y + z * z)

    @staticmethod
    def _mean_std(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        if len(xs) < 2:
            return xs[0], 0.0
        return stats.mean(xs), stats.pstdev(xs)

    def calibrate(self) -> None:
        print(f"[{self.name}] CALIB: keep still {self.CALIB_S:.1f}s ...")

        t0 = time.time()
        gxs, gys, gzs = [], [], []
        dA_samples, jerk_samples, gyro_mag_samples = [], [], []

        ax0, ay0, az0 = self.read_accel_raw()
        last_ax, last_ay, last_az = ax0, ay0, az0

        while time.time() - t0 < self.CALIB_S:
            ax, ay, az = self.read_accel_raw()
            gx, gy, gz = self.read_gyro_raw()

            gxs.append(gx); gys.append(gy); gzs.append(gz)

            d_ax = ax - last_ax
            d_ay = ay - last_ay
            d_az = az - last_az
            dA = self.mag3(d_ax, d_ay, d_az)
            dA_samples.append(dA)

            jerk = dA / max(self.DT, 1e-6)
            jerk_samples.append(jerk)

            gyro_mag_samples.append(self.mag3(gx, gy, gz))

            last_ax, last_ay, last_az = ax, ay, az
            time.sleep(self.DT)

        self.bias_gx = stats.mean(gxs) if gxs else 0.0
        self.bias_gy = stats.mean(gys) if gys else 0.0
        self.bias_gz = stats.mean(gzs) if gzs else 0.0

        dA_mu, dA_sd = self._mean_std(dA_samples)
        jerk_mu, jerk_sd = self._mean_std(jerk_samples)
        gyro_mu, gyro_sd = self._mean_std(gyro_mag_samples)

        self.dA_th = max(self.MIN_dA_TH, dA_mu + self.K_dA * dA_sd)
        self.jerk_th = max(self.MIN_JERK_TH, jerk_mu + self.K_jerk * jerk_sd)
        self.gyro_th = max(self.MIN_GYRO_TH, gyro_mu + self.K_gyro * gyro_sd)

        self.last_ax, self.last_ay, self.last_az = self.read_accel_raw()
        self.last_event_t = 0.0
        self.ready = True

        print(f"[{self.name}] CALIB done. bias=({self.bias_gx:.1f},{self.bias_gy:.1f},{self.bias_gz:.1f}) "
              f"TH(dA={self.dA_th:.1f}, jerk={self.jerk_th:.1f}, gyro={self.gyro_th:.1f})")

    def update(self) -> Optional[Dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("Not calibrated")

        now = time.time()
        ax, ay, az = self.read_accel_raw()
        gx, gy, gz = self.read_gyro_raw()

        gx2 = gx - self.bias_gx
        gy2 = gy - self.bias_gy
        gz2 = gz - self.bias_gz

        d_ax = ax - self.last_ax
        d_ay = ay - self.last_ay
        d_az = az - self.last_az
        dA = self.mag3(d_ax, d_ay, d_az)

        jerk = dA / max(self.DT, 1e-6)
        gyro_mag = self.mag3(gx2, gy2, gz2)

        sig_dA = (dA >= self.dA_th)
        sig_jerk = (jerk >= self.jerk_th)
        sig_gyro = (gyro_mag >= self.gyro_th)

        sig_count = int(sig_dA) + int(sig_jerk) + int(sig_gyro)
        crash = (sig_count >= 2) if self.require_two_signals else (sig_count >= 1)

        if crash and (now - self.last_event_t) >= self.COOLDOWN_S:
            self.last_event_t = now
            ev = {
                "t": now,
                "impact_dA": float(dA),
                "impact_jerk": float(jerk),
                "gyro_mag": float(gyro_mag),
                "signals": {"dA": sig_dA, "jerk": sig_jerk, "gyro": sig_gyro},
            }
            # update last
            self.last_ax, self.last_ay, self.last_az = ax, ay, az
            return ev

        self.last_ax, self.last_ay, self.last_az = ax, ay, az
        return None


# =========================
# Shared state (label yes/no)
# =========================
_lock = threading.Lock()
_collision_label = "no"
_last_yes_t = 0.0
_last_event: Optional[Dict[str, Any]] = None
_last_err_log_t = 0.0


def set_no(reason: str = ""):
    global _collision_label
    with _lock:
        _collision_label = "no"


def set_yes(event: Dict[str, Any]):
    global _collision_label, _last_yes_t, _last_event
    with _lock:
        _collision_label = "yes"
        _last_yes_t = time.time()
        _last_event = event


def get_status() -> Dict[str, Any]:
    with _lock:
        return {
            "label": _collision_label,
            "last_yes_t": _last_yes_t,
            "last_event": _last_event,
        }


# =========================
# Worker thread
# =========================
def detector_worker():
    global _last_err_log_t

    det = CollisionDetector(
        require_two_signals=REQUIRE_TWO_SIGNALS,
        name="SH3001",
    )

    # keep service alive: if calib fails, retry
    while True:
        try:
            det.calibrate()
            break
        except Exception as e:
            now = time.time()
            if now - _last_err_log_t > ERROR_LOG_COOLDOWN:
                print(f"[collision] CALIB ERROR: {e} -> retry in 2s")
                _last_err_log_t = now
            set_no("calib_error")
            time.sleep(2.0)

    # main loop
    while True:
        try:
            ev = det.update()

            if ev is not None:
                set_yes(ev)

            # auto drop back to "no" after YES_HOLD_S
            st = get_status()
            if st["label"] == "yes":
                if (time.time() - st["last_yes_t"]) >= YES_HOLD_S:
                    set_no("timeout")

        except Exception as e:
            # sensor read error -> treat as "no"
            if ERROR_NO_FORCE:
                set_no("sensor_error")

            now = time.time()
            if now - _last_err_log_t > ERROR_LOG_COOLDOWN:
                print(f"[collision] SENSOR ERROR: {e} -> label=no")
                _last_err_log_t = now

            # small backoff
            time.sleep(0.05)

        time.sleep(DT)


# =========================
# Flask API
# =========================
app = Flask(__name__)

@app.get("/colissionstatus")
def colissionstatus():
    """
    Returns collision label 'yes'/'no'
    - default: JSON {"label":"yes"} ...
    - if ?format=text => plain text yes/no
    """
    st = get_status()

    fmt = request.args.get("format", "").lower().strip()
    if fmt == "text":
        return Response(st["label"], mimetype="text/plain")

    return jsonify({
        "ok": True,
        "label": st["label"],
        "last_event": st["last_event"],
    })


def main():
    # start worker thread
    th = threading.Thread(target=detector_worker, daemon=True)
    th.start()

    print(f"[collision] HTTP on http://{HOST}:{PORT}/colissionstatus")
    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()
