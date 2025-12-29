#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smbus2 import SMBus
from time import sleep, time
import math
import statistics as stats
from typing import Optional, Dict, Any, Tuple, List


class CollisionDetector:
    """
    CollisionDetector for SH3001-like IMU (raw I2C registers).

    - Only logs/returns event when collision happens
    - Uses sudden acceleration change (delta-a / jerk) + gyro magnitude as signals
    - Calibrates auto thresholds while robot is still

    NOTE:
    - Raw units are sensor-dependent. This class is for hunting thresholds.
    - Later you can convert to g, m/s^2 if you know sensor scale.
    """

    def __init__(
        self,
        bus_id: int = 1,
        addr: int = 0x36,
        # regs
        reg_ax: int = 0x01,
        reg_ay: int = 0x03,
        reg_az: int = 0x05,
        reg_gx: int = 0x07,
        reg_gy: int = 0x09,
        reg_gz: int = 0x0B,
        # sampling
        dt: float = 0.02,             # 50Hz
        calib_s: float = 3.0,
        cooldown_s: float = 0.8,
        # sensitivity
        k_dA: float = 8.0,            # delta-a threshold = mean + k*std
        k_jerk: float = 8.0,          # jerk threshold = mean + k*std
        k_gyro: float = 8.0,          # gyro magnitude threshold
        # minimum thresholds (anti-noise)
        min_dA_th: float = 30.0,
        min_jerk_th: float = 1500.0,  # note: jerk is dA/DT, so typical number bigger
        min_gyro_th: float = 10.0,
        # event rule
        require_two_signals: bool = False,  # True = phải có >=2 điều kiện mới crash
        print_events: bool = True,
        name: str = "IMU",
    ):
        self.bus_id = bus_id
        self.addr = addr

        self.REG_AX = reg_ax
        self.REG_AY = reg_ay
        self.REG_AZ = reg_az
        self.REG_GX = reg_gx
        self.REG_GY = reg_gy
        self.REG_GZ = reg_gz

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
        self.print_events = print_events
        self.name = name

        self.bus = SMBus(self.bus_id)

        # state
        self.bias_gx = 0.0
        self.bias_gy = 0.0
        self.bias_gz = 0.0

        self.dA_th = self.MIN_dA_TH
        self.jerk_th = self.MIN_JERK_TH
        self.gyro_th = self.MIN_GYRO_TH

        self.last_ax = None
        self.last_ay = None
        self.last_az = None
        self.last_t = None
        self.last_event_t = 0.0

        self.ready = False

    # --------------------------
    # Low-level IMU read helpers
    # --------------------------
    def _read_word(self, reg: int) -> int:
        hi = self.bus.read_byte_data(self.addr, reg)
        lo = self.bus.read_byte_data(self.addr, reg + 1)
        val = (hi << 8) | lo
        if val & 0x8000:
            val -= 65536
        return val

    def read_accel_raw(self) -> Tuple[int, int, int]:
        ax = self._read_word(self.REG_AX)
        ay = self._read_word(self.REG_AY)
        az = self._read_word(self.REG_AZ)
        return ax, ay, az

    def read_gyro_raw(self) -> Tuple[int, int, int]:
        gx = self._read_word(self.REG_GX)
        gy = self._read_word(self.REG_GY)
        gz = self._read_word(self.REG_GZ)
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

    # --------------------------
    # Calibrate thresholds
    # --------------------------
    def calibrate(self) -> None:
        """
        Keep robot still for CALIB_S seconds.
        Estimates gyro bias and noise distribution to auto-set thresholds.
        """
        print(f"[{self.name}] CALIB: keep still {self.CALIB_S:.1f}s (dt={self.DT}s) ...")

        t0 = time()

        gxs, gys, gzs = [], [], []
        dA_samples = []
        jerk_samples = []
        gyro_mag_samples = []

        ax0, ay0, az0 = self.read_accel_raw()
        last_ax, last_ay, last_az = ax0, ay0, az0

        while time() - t0 < self.CALIB_S:
            ax, ay, az = self.read_accel_raw()
            gx, gy, gz = self.read_gyro_raw()

            gxs.append(gx)
            gys.append(gy)
            gzs.append(gz)

            d_ax = ax - last_ax
            d_ay = ay - last_ay
            d_az = az - last_az
            dA = self.mag3(d_ax, d_ay, d_az)
            dA_samples.append(dA)

            jerk = dA / max(self.DT, 1e-6)
            jerk_samples.append(jerk)

            gyro_mag_samples.append(self.mag3(gx, gy, gz))

            last_ax, last_ay, last_az = ax, ay, az
            sleep(self.DT)

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
        self.last_t = time()
        self.last_event_t = 0.0
        self.ready = True

        print(f"[{self.name}] CALIB done.")
        print(f"  gyro_bias=({self.bias_gx:.1f},{self.bias_gy:.1f},{self.bias_gz:.1f})")
        print(f"  TH_dA   ={self.dA_th:.1f}   (mean={dA_mu:.1f}, sd={dA_sd:.1f}, K={self.K_dA})")
        print(f"  TH_jerk ={self.jerk_th:.1f} (mean={jerk_mu:.1f}, sd={jerk_sd:.1f}, K={self.K_jerk})")
        print(f"  TH_gyro ={self.gyro_th:.1f} (mean={gyro_mu:.1f}, sd={gyro_sd:.1f}, K={self.K_gyro})")
        print(f"  rule: require_two_signals={self.require_two_signals}, cooldown={self.COOLDOWN_S}s\n")

    # --------------------------
    # Event detection
    # --------------------------
    def _axis_label(self, vx: float, vy: float, vz: float) -> Tuple[str, float]:
        """
        Returns dominant axis direction string + dominance ratio.
        Example: "+X" or "-Y"
        """
        ax = abs(vx); ay = abs(vy); az = abs(vz)
        m = max(ax, ay, az, 1e-9)

        if m == ax:
            return ("+X" if vx >= 0 else "-X"), ax / (ax + ay + az + 1e-9)
        if m == ay:
            return ("+Y" if vy >= 0 else "-Y"), ay / (ax + ay + az + 1e-9)
        return ("+Z" if vz >= 0 else "-Z"), az / (ax + ay + az + 1e-9)

    def update(self) -> Optional[Dict[str, Any]]:
        """
        Read one sample; return event dict when collision happens, else None.
        """
        if not self.ready:
            raise RuntimeError("CollisionDetector not calibrated. Call calibrate() first.")

        now = time()
        ax, ay, az = self.read_accel_raw()
        gx, gy, gz = self.read_gyro_raw()

        # remove gyro bias
        gx2 = gx - self.bias_gx
        gy2 = gy - self.bias_gy
        gz2 = gz - self.bias_gz

        # delta acceleration (raw)
        d_ax = ax - self.last_ax
        d_ay = ay - self.last_ay
        d_az = az - self.last_az
        dA = self.mag3(d_ax, d_ay, d_az)

        # jerk (raw per second)
        jerk = dA / max(self.DT, 1e-6)

        gyro_mag = self.mag3(gx2, gy2, gz2)

        # signals
        sig_dA = (dA >= self.dA_th)
        sig_jerk = (jerk >= self.jerk_th)
        sig_gyro = (gyro_mag >= self.gyro_th)

        # decision
        sig_count = int(sig_dA) + int(sig_jerk) + int(sig_gyro)
        crash = (sig_count >= 2) if self.require_two_signals else (sig_count >= 1)

        # cooldown
        if crash and (now - self.last_event_t) >= self.COOLDOWN_S:
            axis_dir, dominance = self._axis_label(d_ax, d_ay, d_az)
            # “force intensity” proxy:
            # - impact_dA: how strong acceleration changed per sample
            # - impact_jerk: per second
            event = {
                "t": now,
                "axis_dir": axis_dir,               # dominant change direction (+X/-X/+Y/-Y/+Z/-Z)
                "dominance": dominance,             # 0..1 ratio (how dominant that axis is)
                "dA_vec": (d_ax, d_ay, d_az),       # raw delta accel vector
                "impact_dA": float(dA),             # raw magnitude per sample
                "impact_jerk": float(jerk),         # raw magnitude per second
                "gyro_mag": float(gyro_mag),
                "signals": {
                    "dA": sig_dA,
                    "jerk": sig_jerk,
                    "gyro": sig_gyro,
                },
                "thresholds": {
                    "dA_th": float(self.dA_th),
                    "jerk_th": float(self.jerk_th),
                    "gyro_th": float(self.gyro_th),
                },
            }

            self.last_event_t = now

            if self.print_events:
                sigs = []
                if sig_dA: sigs.append("dA")
                if sig_jerk: sigs.append("jerk")
                if sig_gyro: sigs.append("gyro")

                print(
                    f"[{self.name}] COLLISION "
                    f"t={event['t']:.3f}  dir={axis_dir} (dom={dominance:.2f})  "
                    f"impact_dA={event['impact_dA']:.1f}  impact_jerk={event['impact_jerk']:.1f}  "
                    f"gyro_mag={event['gyro_mag']:.1f}  signals={'+'.join(sigs)}"
                )

            # update last sample
            self.last_ax, self.last_ay, self.last_az = ax, ay, az
            self.last_t = now
            return event

        # update last sample
        self.last_ax, self.last_ay, self.last_az = ax, ay, az
        self.last_t = now
        return None

    def loop(self) -> None:
        """
        Simple loop for testing: only prints when collision occurs.
        """
        if not self.ready:
            self.calibrate()

        while True:
            try:
                _ = self.update()
            except Exception as e:
                print(f"[{self.name}] ERROR: {e}")
            sleep(self.DT)


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    det = CollisionDetector(
        bus_id=1,
        addr=0x36,
        dt=0.02,
        calib_s=3.0,
        cooldown_s=0.8,
        # sensitivity: tune later
        k_dA=8.0,
        k_jerk=8.0,
        k_gyro=8.0,
        min_dA_th=30.0,
        min_jerk_th=1500.0,
        min_gyro_th=10.0,
        require_two_signals=True,  # True nếu bạn muốn “chắc ăn hơn”
        print_events=True,
        name="SH3001",
    )
    det.loop()
