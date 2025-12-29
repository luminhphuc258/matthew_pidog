#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smbus2 import SMBus
from time import sleep, time
import math
import statistics as stats
from collections import deque
from typing import Optional, Dict, Any, Tuple, List


class CollisionDetector:
    """
    Anti-false-positive collision detector (good for walking robots).

    Key changes vs "too sensitive" version:
    - Adaptive thresholds using rolling robust stats (median + K * MAD)
    - Impulse (spike) condition: must rise suddenly across threshold
    - Require >=2 signals by default: jerk + (dA or gyro)

    It only prints/logs when COLLISION happens.
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
        dt: float = 0.02,           # 50Hz
        calib_s: float = 3.0,
        cooldown_s: float = 1.0,
        # rolling window for adaptive thresholds
        win_s: float = 1.0,         # 1.0s window (recommend 0.8~1.5s)
        # robust K (bigger = less sensitive)
        k_dA: float = 10.0,
        k_jerk: float = 10.0,
        k_gyro: float = 10.0,
        # minimum thresholds (anti-noise floor)
        min_dA_th: float = 60.0,
        min_jerk_th: float = 2500.0,
        min_gyro_th: float = 25.0,
        # impulse condition
        spike_ratio_prev: float = 0.70,  # prev must be < 0.70*TH
        rearm_ratio: float = 0.45,       # must drop below 0.45*TH to re-arm
        # decision rule
        require_two_signals: bool = True,
        print_events: bool = True,
        name: str = "SH3001",
    ):
        self.bus_id = bus_id
        self.addr = addr

        self.REG_AX = reg_ax
        self.REG_AY = reg_ay
        self.REG_AZ = reg_az
        self.REG_GX = reg_gx
        self.REG_GY = reg_gy
        self.REG_GZ = reg_gz

        self.DT = float(dt)
        self.CALIB_S = float(calib_s)
        self.COOLDOWN_S = float(cooldown_s)

        self.win_s = float(win_s)
        self.win_n = max(10, int(self.win_s / max(self.DT, 1e-6)))  # at least 10 samples

        self.K_dA = float(k_dA)
        self.K_jerk = float(k_jerk)
        self.K_gyro = float(k_gyro)

        self.MIN_dA_TH = float(min_dA_th)
        self.MIN_JERK_TH = float(min_jerk_th)
        self.MIN_GYRO_TH = float(min_gyro_th)

        self.spike_ratio_prev = float(spike_ratio_prev)
        self.rearm_ratio = float(rearm_ratio)

        self.require_two_signals = bool(require_two_signals)
        self.print_events = bool(print_events)
        self.name = name

        self.bus = SMBus(self.bus_id)

        # biases
        self.bias_gx = 0.0
        self.bias_gy = 0.0
        self.bias_gz = 0.0

        # state
        self.last_ax = None
        self.last_ay = None
        self.last_az = None
        self.prev_dA = 0.0
        self.prev_gyro = 0.0
        self.last_event_t = 0.0

        # rolling buffers (adaptive thresholds)
        self.buf_dA = deque(maxlen=self.win_n)
        self.buf_jerk = deque(maxlen=self.win_n)
        self.buf_gyro = deque(maxlen=self.win_n)

        # re-arm flags (avoid retrigger during continuous shaking)
        self.armed = True

        self.ready = False

    # --------------------------
    # Low-level helpers
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
    def _median(xs: List[float]) -> float:
        return stats.median(xs) if xs else 0.0

    @staticmethod
    def _mad(xs: List[float], med: float) -> float:
        """Median Absolute Deviation."""
        if not xs:
            return 0.0
        devs = [abs(x - med) for x in xs]
        return stats.median(devs) if devs else 0.0

    def _robust_threshold(self, xs: deque, k: float, min_th: float) -> float:
        """
        Robust threshold: median + k * (1.4826 * MAD)
        1.4826 converts MAD to std-like scale under normal noise.
        """
        if len(xs) < 8:
            return float(min_th)
        arr = list(xs)
        med = self._median(arr)
        mad = self._mad(arr, med)
        sigma = 1.4826 * mad
        th = med + k * sigma
        return float(max(min_th, th))

    # --------------------------
    # Calibration
    # --------------------------
    def calibrate(self) -> None:
        """
        Keep still to estimate gyro bias.
        (Thresholds are adaptive later, so we mainly need bias.)
        """
        print(f"[{self.name}] CALIB: keep still {self.CALIB_S:.1f}s (dt={self.DT}s) ...")

        t0 = time()
        gxs, gys, gzs = [], [], []

        # init last accel
        ax0, ay0, az0 = self.read_accel_raw()
        self.last_ax, self.last_ay, self.last_az = ax0, ay0, az0
        self.prev_dA = 0.0
        self.prev_gyro = 0.0

        while time() - t0 < self.CALIB_S:
            gx, gy, gz = self.read_gyro_raw()
            gxs.append(gx); gys.append(gy); gzs.append(gz)
            sleep(self.DT)

        self.bias_gx = stats.mean(gxs) if gxs else 0.0
        self.bias_gy = stats.mean(gys) if gys else 0.0
        self.bias_gz = stats.mean(gzs) if gzs else 0.0

        # Warm-up buffers a bit (so adaptive TH starts stable)
        self.buf_dA.clear()
        self.buf_jerk.clear()
        self.buf_gyro.clear()

        for _ in range(self.win_n):
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

            self.buf_dA.append(dA)
            self.buf_jerk.append(jerk)
            self.buf_gyro.append(gyro_mag)

            self.last_ax, self.last_ay, self.last_az = ax, ay, az
            self.prev_dA = dA
            self.prev_gyro = gyro_mag
            sleep(self.DT)

        self.last_event_t = 0.0
        self.armed = True
        self.ready = True

        # show current estimated thresholds at end of calib
        th_dA = self._robust_threshold(self.buf_dA, self.K_dA, self.MIN_dA_TH)
        th_jerk = self._robust_threshold(self.buf_jerk, self.K_jerk, self.MIN_JERK_TH)
        th_gyro = self._robust_threshold(self.buf_gyro, self.K_gyro, self.MIN_GYRO_TH)

        print(f"[{self.name}] CALIB done.")
        print(f"  gyro_bias=({self.bias_gx:.1f},{self.bias_gy:.1f},{self.bias_gz:.1f})")
        print(f"  initial adaptive TH: dA={th_dA:.1f}  jerk={th_jerk:.1f}  gyro={th_gyro:.1f}")
        print(f"  rule: require_two_signals={self.require_two_signals}, cooldown={self.COOLDOWN_S}s, window={self.win_n} samples\n")

    # --------------------------
    # Direction label
    # --------------------------
    def _axis_label(self, vx: float, vy: float, vz: float) -> Tuple[str, float]:
        ax = abs(vx); ay = abs(vy); az = abs(vz)
        s = ax + ay + az + 1e-9
        m = max(ax, ay, az, 1e-9)
        if m == ax:
            return ("+X" if vx >= 0 else "-X"), ax / s
        if m == ay:
            return ("+Y" if vy >= 0 else "-Y"), ay / s
        return ("+Z" if vz >= 0 else "-Z"), az / s

    # --------------------------
    # Update / detect
    # --------------------------
    def update(self) -> Optional[Dict[str, Any]]:
        """
        Returns event dict only when collision is detected, else None.
        """
        if not self.ready:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        now = time()

        ax, ay, az = self.read_accel_raw()
        gx, gy, gz = self.read_gyro_raw()

        # remove gyro bias
        gx2 = gx - self.bias_gx
        gy2 = gy - self.bias_gy
        gz2 = gz - self.bias_gz

        # delta accel
        d_ax = ax - self.last_ax
        d_ay = ay - self.last_ay
        d_az = az - self.last_az
        dA = self.mag3(d_ax, d_ay, d_az)
        jerk = dA / max(self.DT, 1e-6)
        gyro_mag = self.mag3(gx2, gy2, gz2)

        # push to buffers (adaptive)
        self.buf_dA.append(dA)
        self.buf_jerk.append(jerk)
        self.buf_gyro.append(gyro_mag)

        # compute adaptive thresholds
        th_dA = self._robust_threshold(self.buf_dA, self.K_dA, self.MIN_dA_TH)
        th_jerk = self._robust_threshold(self.buf_jerk, self.K_jerk, self.MIN_JERK_TH)
        th_gyro = self._robust_threshold(self.buf_gyro, self.K_gyro, self.MIN_GYRO_TH)

        # impulse (spike) gating: must rise across threshold (avoid continuous movement)
        spike_dA = (dA >= th_dA) and (self.prev_dA < self.spike_ratio_prev * th_dA)
        spike_gyro = (gyro_mag >= th_gyro) and (self.prev_gyro < self.spike_ratio_prev * th_gyro)
        spike_jerk = (jerk >= th_jerk)  # jerk already "spiky" by nature

        # signals
        sig_dA = spike_dA
        sig_gyro = spike_gyro
        sig_jerk = spike_jerk

        # re-arm logic: after a trigger, wait until signals drop
        if not self.armed:
            if (dA < self.rearm_ratio * th_dA) and (gyro_mag < self.rearm_ratio * th_gyro):
                self.armed = True

        # decision: prefer jerk + (dA or gyro) to avoid walking noise
        # If you want even stricter: require jerk + dA + gyro (3 signals)
        if self.require_two_signals:
            crash = self.armed and (sig_jerk and (sig_dA or sig_gyro))
        else:
            crash = self.armed and (sig_jerk or sig_dA or sig_gyro)

        # cooldown
        if crash and (now - self.last_event_t) >= self.COOLDOWN_S:
            axis_dir, dominance = self._axis_label(d_ax, d_ay, d_az)
            event = {
                "t": now,
                "axis_dir": axis_dir,
                "dominance": dominance,
                "dA_vec": (d_ax, d_ay, d_az),
                "impact_dA": float(dA),
                "impact_jerk": float(jerk),
                "gyro_mag": float(gyro_mag),
                "signals": {"dA_spike": sig_dA, "jerk": sig_jerk, "gyro_spike": sig_gyro},
                "thresholds": {"dA_th": float(th_dA), "jerk_th": float(th_jerk), "gyro_th": float(th_gyro)},
            }

            self.last_event_t = now
            self.armed = False  # disarm until re-armed

            if self.print_events:
                sigs = []
                if sig_jerk: sigs.append("jerk")
                if sig_dA: sigs.append("dA_spike")
                if sig_gyro: sigs.append("gyro_spike")
                print(
                    f"[{self.name}] COLLISION "
                    f"t={now:.3f} dir={axis_dir}(dom={dominance:.2f}) "
                    f"impact_dA={dA:.1f} impact_jerk={jerk:.1f} gyro={gyro_mag:.1f} "
                    f"TH(dA={th_dA:.1f}, jerk={th_jerk:.1f}, gyro={th_gyro:.1f}) "
                    f"signals={'+'.join(sigs)}"
                )

            # update state
            self.last_ax, self.last_ay, self.last_az = ax, ay, az
            self.prev_dA = dA
            self.prev_gyro = gyro_mag
            return event

        # update state
        self.last_ax, self.last_ay, self.last_az = ax, ay, az
        self.prev_dA = dA
        self.prev_gyro = gyro_mag
        return None

    def loop(self) -> None:
        if not self.ready:
            self.calibrate()

        while True:
            try:
                _ = self.update()  # only logs when collision
            except Exception as e:
                print(f"[{self.name}] ERROR: {e}")
            sleep(self.DT)


# --------------------------
# Run directly
# --------------------------
if __name__ == "__main__":
    det = CollisionDetector(
        bus_id=1,
        addr=0x36,
        dt=0.02,
        calib_s=3.0,
        cooldown_s=1.0,
        win_s=1.0,

        # less sensitive when moving:
        k_dA=10.0,
        k_jerk=10.0,
        k_gyro=10.0,

        # raise mins to avoid walking noise:
        min_dA_th=60.0,
        min_jerk_th=2500.0,
        min_gyro_th=25.0,

        # spike gating
        spike_ratio_prev=0.70,
        rearm_ratio=0.45,

        # safest rule for moving robot
        require_two_signals=True,

        print_events=True,
        name="SH3001",
    )
    det.loop()
