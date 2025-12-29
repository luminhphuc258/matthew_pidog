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
    Collision Detector (robust, anti-false-positive even when standing still)

    Improvements:
    - EMA smoothing on accel (reduce noise spikes)
    - Confirmation window: need >= confirm_count hits within confirm_win samples
    - STILL/MOVE mode:
        * detect stillness from gyro + dA medians
        * in STILL -> thresholds multiplied (stricter) to prevent false triggers
    - Adaptive thresholds via robust stats: median + K*(1.4826*MAD)
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
        dt: float = 0.02,
        calib_s: float = 4.0,
        cooldown_s: float = 1.2,
        # adaptive window
        win_s: float = 1.2,
        # robust K (bigger = less sensitive)
        k_dA: float = 12.0,
        k_jerk: float = 12.0,
        k_gyro: float = 12.0,
        # minimum thresholds (anti-noise floor)
        min_dA_th: float = 90.0,
        min_jerk_th: float = 4000.0,
        min_gyro_th: float = 30.0,
        # spike gating
        spike_ratio_prev: float = 0.80,
        rearm_ratio: float = 0.45,
        # confirmation
        confirm_win: int = 3,      # lookback samples
        confirm_count: int = 2,    # need >=2 hits in that window
        # still mode tightening
        still_mult: float = 1.7,   # multiply thresholds when robot considered "still"
        still_detect_k: float = 3.0,  # stillness boundary = med + still_detect_k*sigma
        still_hold_s: float = 0.4, # must be still for this long to enter STILL
        # detection rule
        require_two_signals: bool = True,  # jerk + (dA or gyro) is safest
        print_events: bool = True,
        name: str = "SH3001",
        # EMA smoothing (0..1). lower = more smoothing.
        ema_alpha: float = 0.35,
    ):
        self.bus = SMBus(bus_id)
        self.addr = addr

        self.REG_AX, self.REG_AY, self.REG_AZ = reg_ax, reg_ay, reg_az
        self.REG_GX, self.REG_GY, self.REG_GZ = reg_gx, reg_gy, reg_gz

        self.DT = float(dt)
        self.CALIB_S = float(calib_s)
        self.COOLDOWN_S = float(cooldown_s)

        self.win_n = max(12, int(float(win_s) / max(self.DT, 1e-6)))
        self.K_dA, self.K_jerk, self.K_gyro = float(k_dA), float(k_jerk), float(k_gyro)

        self.MIN_dA_TH = float(min_dA_th)
        self.MIN_JERK_TH = float(min_jerk_th)
        self.MIN_GYRO_TH = float(min_gyro_th)

        self.spike_ratio_prev = float(spike_ratio_prev)
        self.rearm_ratio = float(rearm_ratio)

        self.confirm_win = int(max(1, confirm_win))
        self.confirm_count = int(max(1, confirm_count))

        self.still_mult = float(still_mult)
        self.still_detect_k = float(still_detect_k)
        self.still_hold_n = max(1, int(float(still_hold_s) / max(self.DT, 1e-6)))

        self.require_two_signals = bool(require_two_signals)
        self.print_events = bool(print_events)
        self.name = name

        self.ema_alpha = float(ema_alpha)
        if not (0.01 <= self.ema_alpha <= 0.99):
            self.ema_alpha = 0.35

        # bias
        self.bias_gx = 0.0
        self.bias_gy = 0.0
        self.bias_gz = 0.0

        # state
        self.ready = False
        self.last_event_t = 0.0
        self.armed = True

        self.prev_dA = 0.0
        self.prev_gyro = 0.0

        # last raw accel (for delta)
        self.last_ax = None
        self.last_ay = None
        self.last_az = None

        # EMA accel values (smoothed)
        self.ema_ax = None
        self.ema_ay = None
        self.ema_az = None

        # rolling buffers
        self.buf_dA = deque(maxlen=self.win_n)
        self.buf_jerk = deque(maxlen=self.win_n)
        self.buf_gyro = deque(maxlen=self.win_n)

        # confirmation buffer
        self.cand_hits = deque(maxlen=self.confirm_win)

        # stillness tracking
        self.still_counter = 0
        self.mode = "MOVE"  # or "STILL"

    # ---------------- low-level ----------------
    def _read_word(self, reg: int) -> int:
        hi = self.bus.read_byte_data(self.addr, reg)
        lo = self.bus.read_byte_data(self.addr, reg + 1)
        val = (hi << 8) | lo
        if val & 0x8000:
            val -= 65536
        return val

    def read_accel_raw(self) -> Tuple[int, int, int]:
        return (
            self._read_word(self.REG_AX),
            self._read_word(self.REG_AY),
            self._read_word(self.REG_AZ),
        )

    def read_gyro_raw(self) -> Tuple[int, int, int]:
        return (
            self._read_word(self.REG_GX),
            self._read_word(self.REG_GY),
            self._read_word(self.REG_GZ),
        )

    @staticmethod
    def mag3(x: float, y: float, z: float) -> float:
        return math.sqrt(x * x + y * y + z * z)

    @staticmethod
    def _median(xs: List[float]) -> float:
        return stats.median(xs) if xs else 0.0

    @staticmethod
    def _mad(xs: List[float], med: float) -> float:
        if not xs:
            return 0.0
        devs = [abs(x - med) for x in xs]
        return stats.median(devs) if devs else 0.0

    def _robust_stats(self, xs: deque) -> Tuple[float, float]:
        """
        returns (median, sigma_est) where sigma_est ~ std using MAD
        """
        if len(xs) < 8:
            return 0.0, 0.0
        arr = list(xs)
        med = self._median(arr)
        mad = self._mad(arr, med)
        sigma = 1.4826 * mad
        return float(med), float(sigma)

    def _robust_threshold(self, xs: deque, k: float, min_th: float) -> float:
        med, sigma = self._robust_stats(xs)
        th = med + k * sigma
        return float(max(min_th, th))

    def _axis_label(self, vx: float, vy: float, vz: float) -> Tuple[str, float]:
        ax = abs(vx); ay = abs(vy); az = abs(vz)
        s = ax + ay + az + 1e-9
        m = max(ax, ay, az, 1e-9)
        if m == ax:
            return ("+X" if vx >= 0 else "-X"), ax / s
        if m == ay:
            return ("+Y" if vy >= 0 else "-Y"), ay / s
        return ("+Z" if vz >= 0 else "-Z"), az / s

    # ---------------- calibration ----------------
    def calibrate(self) -> None:
        print(f"[{self.name}] CALIB: keep still {self.CALIB_S:.1f}s (dt={self.DT}s) ...")
        t0 = time()
        gxs, gys, gzs = [], [], []

        ax0, ay0, az0 = self.read_accel_raw()
        self.last_ax, self.last_ay, self.last_az = ax0, ay0, az0

        # init EMA
        self.ema_ax, self.ema_ay, self.ema_az = float(ax0), float(ay0), float(az0)

        while time() - t0 < self.CALIB_S:
            gx, gy, gz = self.read_gyro_raw()
            gxs.append(gx); gys.append(gy); gzs.append(gz)
            sleep(self.DT)

        self.bias_gx = stats.mean(gxs) if gxs else 0.0
        self.bias_gy = stats.mean(gys) if gys else 0.0
        self.bias_gz = stats.mean(gzs) if gzs else 0.0

        # warm buffers
        self.buf_dA.clear(); self.buf_jerk.clear(); self.buf_gyro.clear()
        self.cand_hits.clear()

        self.prev_dA = 0.0
        self.prev_gyro = 0.0

        for _ in range(self.win_n):
            ax, ay, az = self.read_accel_raw()
            gx, gy, gz = self.read_gyro_raw()
            gx2 = gx - self.bias_gx
            gy2 = gy - self.bias_gy
            gz2 = gz - self.bias_gz

            # EMA smoothing for accel
            a = self.ema_alpha
            self.ema_ax = (1 - a) * self.ema_ax + a * float(ax)
            self.ema_ay = (1 - a) * self.ema_ay + a * float(ay)
            self.ema_az = (1 - a) * self.ema_az + a * float(az)

            d_ax = self.ema_ax - float(self.last_ax)
            d_ay = self.ema_ay - float(self.last_ay)
            d_az = self.ema_az - float(self.last_az)

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
        self.still_counter = 0
        self.mode = "MOVE"
        self.ready = True

        th_dA = self._robust_threshold(self.buf_dA, self.K_dA, self.MIN_dA_TH)
        th_jerk = self._robust_threshold(self.buf_jerk, self.K_jerk, self.MIN_JERK_TH)
        th_gyro = self._robust_threshold(self.buf_gyro, self.K_gyro, self.MIN_GYRO_TH)

        print(f"[{self.name}] CALIB done.")
        print(f"  gyro_bias=({self.bias_gx:.1f},{self.bias_gy:.1f},{self.bias_gz:.1f})")
        print(f"  initial TH: dA={th_dA:.1f} jerk={th_jerk:.1f} gyro={th_gyro:.1f}")
        print(f"  confirm: {self.confirm_count}/{self.confirm_win} | STILL mult={self.still_mult} | EMA alpha={self.ema_alpha}\n")

    # ---------------- update ----------------
    def update(self) -> Optional[Dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        now = time()

        ax, ay, az = self.read_accel_raw()
        gx, gy, gz = self.read_gyro_raw()

        gx2 = gx - self.bias_gx
        gy2 = gy - self.bias_gy
        gz2 = gz - self.bias_gz
        gyro_mag = self.mag3(gx2, gy2, gz2)

        # EMA smoothing accel
        a = self.ema_alpha
        self.ema_ax = (1 - a) * self.ema_ax + a * float(ax)
        self.ema_ay = (1 - a) * self.ema_ay + a * float(ay)
        self.ema_az = (1 - a) * self.ema_az + a * float(az)

        # delta using EMA accel vs last raw accel (works well for spike noise)
        d_ax = self.ema_ax - float(self.last_ax)
        d_ay = self.ema_ay - float(self.last_ay)
        d_az = self.ema_az - float(self.last_az)

        dA = self.mag3(d_ax, d_ay, d_az)
        jerk = dA / max(self.DT, 1e-6)

        # update rolling buffers
        self.buf_dA.append(dA)
        self.buf_jerk.append(jerk)
        self.buf_gyro.append(gyro_mag)

        # base thresholds
        th_dA = self._robust_threshold(self.buf_dA, self.K_dA, self.MIN_dA_TH)
        th_jerk = self._robust_threshold(self.buf_jerk, self.K_jerk, self.MIN_JERK_TH)
        th_gyro = self._robust_threshold(self.buf_gyro, self.K_gyro, self.MIN_GYRO_TH)

        # STILL/MOVE detect (robust)
        dA_med, dA_sig = self._robust_stats(self.buf_dA)
        g_med, g_sig = self._robust_stats(self.buf_gyro)
        still_dA_limit = dA_med + self.still_detect_k * dA_sig
        still_g_limit = g_med + self.still_detect_k * g_sig

        if (dA <= max(self.MIN_dA_TH, still_dA_limit)) and (gyro_mag <= max(self.MIN_GYRO_TH, still_g_limit)):
            self.still_counter += 1
        else:
            self.still_counter = 0

        self.mode = "STILL" if self.still_counter >= self.still_hold_n else "MOVE"

        # tighten thresholds if STILL
        if self.mode == "STILL":
            th_dA *= self.still_mult
            th_jerk *= self.still_mult
            th_gyro *= self.still_mult

        # spike gating
        spike_dA = (dA >= th_dA) and (self.prev_dA < self.spike_ratio_prev * th_dA)
        spike_gyro = (gyro_mag >= th_gyro) and (self.prev_gyro < self.spike_ratio_prev * th_gyro)
        spike_jerk = (jerk >= th_jerk)  # jerk itself is spike-like

        # signals
        sig_dA = spike_dA
        sig_gyro = spike_gyro
        sig_jerk = spike_jerk

        # re-arm logic
        if not self.armed:
            if (dA < self.rearm_ratio * th_dA) and (gyro_mag < self.rearm_ratio * th_gyro):
                self.armed = True

        # candidate condition (before confirm)
        if self.require_two_signals:
            candidate = self.armed and (sig_jerk and (sig_dA or sig_gyro))
        else:
            candidate = self.armed and (sig_jerk or sig_dA or sig_gyro)

        # confirmation window
        self.cand_hits.append(1 if candidate else 0)
        confirmed = (sum(self.cand_hits) >= self.confirm_count)

        # final decision + cooldown
        if confirmed and (now - self.last_event_t) >= self.COOLDOWN_S:
            axis_dir, dominance = self._axis_label(d_ax, d_ay, d_az)

            event = {
                "t": now,
                "mode": self.mode,
                "axis_dir": axis_dir,
                "dominance": dominance,
                "dA_vec": (float(d_ax), float(d_ay), float(d_az)),
                "impact_dA": float(dA),
                "impact_jerk": float(jerk),
                "gyro_mag": float(gyro_mag),
                "signals": {"jerk": sig_jerk, "dA_spike": sig_dA, "gyro_spike": sig_gyro},
                "thresholds": {"dA_th": float(th_dA), "jerk_th": float(th_jerk), "gyro_th": float(th_gyro)},
                "confirm": {"hits": int(sum(self.cand_hits)), "need": self.confirm_count, "win": self.confirm_win},
            }

            self.last_event_t = now
            self.armed = False
            self.cand_hits.clear()

            if self.print_events:
                sigs = []
                if sig_jerk: sigs.append("jerk")
                if sig_dA: sigs.append("dA_spike")
                if sig_gyro: sigs.append("gyro_spike")

                print(
                    f"[{self.name}] COLLISION mode={self.mode} "
                    f"t={now:.3f} dir={axis_dir}(dom={dominance:.2f}) "
                    f"impact_dA={dA:.1f} impact_jerk={jerk:.1f} gyro={gyro_mag:.1f} "
                    f"TH(dA={th_dA:.1f}, jerk={th_jerk:.1f}, gyro={th_gyro:.1f}) "
                    f"confirm={event['confirm']['hits']}/{self.confirm_win} "
                    f"signals={'+'.join(sigs)}"
                )

            # update prev + last accel
            self.last_ax, self.last_ay, self.last_az = ax, ay, az
            self.prev_dA = dA
            self.prev_gyro = gyro_mag
            return event

        # update prev + last accel
        self.last_ax, self.last_ay, self.last_az = ax, ay, az
        self.prev_dA = dA
        self.prev_gyro = gyro_mag
        return None

    def loop(self) -> None:
        if not self.ready:
            self.calibrate()

        while True:
            try:
                self.update()  # only prints when collision
            except Exception as e:
                print(f"[{self.name}] ERROR: {e}")
            sleep(self.DT)


if __name__ == "__main__":
    det = CollisionDetector(
        bus_id=1,
        addr=0x36,
        dt=0.02,
        calib_s=4.0,
        cooldown_s=1.2,

        win_s=1.2,
        k_dA=12.0,
        k_jerk=12.0,
        k_gyro=12.0,

        min_dA_th=90.0,
        min_jerk_th=4000.0,
        min_gyro_th=30.0,

        spike_ratio_prev=0.80,
        rearm_ratio=0.45,

        confirm_win=3,
        confirm_count=2,

        still_mult=1.7,
        still_detect_k=3.0,
        still_hold_s=0.4,

        require_two_signals=True,
        ema_alpha=0.35,

        print_events=True,
        name="SH3001",
    )
    det.loop()
