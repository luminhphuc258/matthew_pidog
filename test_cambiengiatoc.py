#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smbus2 import SMBus
from time import sleep, time
import math
import statistics as stats

ADDR_ACC = 0x36
BUS_ID = 1

REG_AX = 0x01
REG_AY = 0x03
REG_AZ = 0x05
REG_GX = 0x07
REG_GY = 0x09
REG_GZ = 0x0B

# sampling
DT = 0.02          # 50 Hz (quan trọng để bắt va chạm)
CALIB_S = 3.0      # đứng yên để calibrate
COOLDOWN_S = 0.8   # tránh spam CRASH liên tục

# sensitivity (tăng/giảm nếu muốn)
K_JERK = 8.0       # jerk threshold = mean + K*std
K_GYRO = 8.0
K_ACCDELTA = 6.0

# minimum thresholds (để khỏi quá nhạy nếu noise thấp)
MIN_JERK_TH = 40.0
MIN_GYRO_TH = 10.0
MIN_ACCDELTA_TH = 30.0

bus = SMBus(BUS_ID)

def read_word(addr, reg):
    hi = bus.read_byte_data(addr, reg)
    lo = bus.read_byte_data(addr, reg + 1)
    val = (hi << 8) | lo
    if val & 0x8000:
        val -= 65536
    return val

def read_accel_raw():
    ax = read_word(ADDR_ACC, REG_AX)
    ay = read_word(ADDR_ACC, REG_AY)
    az = read_word(ADDR_ACC, REG_AZ)
    return ax, ay, az

def read_gyro_raw():
    gx = read_word(ADDR_ACC, REG_GX)
    gy = read_word(ADDR_ACC, REG_GY)
    gz = read_word(ADDR_ACC, REG_GZ)
    return gx, gy, gz

def mag3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def mean_std(xs):
    if len(xs) < 2:
        return (xs[0] if xs else 0.0), 0.0
    return stats.mean(xs), stats.pstdev(xs)

def main():
    print("=== IMU SH3001 CRASH TEST (AUTO THRESHOLD) ===")
    print(f"I2C addr=0x{ADDR_ACC:02X} bus={BUS_ID}  dt={DT}s")

    # quick read check
    try:
        ax, ay, az = read_accel_raw()
        gx, gy, gz = read_gyro_raw()
        print(f"[BOOT] ACC=({ax},{ay},{az}) GYRO=({gx},{gy},{gz})")
    except Exception as e:
        print("[BOOT] ERROR reading IMU:", e)
        return

    print(f"\n[CALIB] Giữ robot đứng yên {CALIB_S:.1f}s để auto set ngưỡng...")
    t0 = time()

    jerk_samples = []
    gyro_samples = []
    accmag_samples = []
    accdelta_samples = []

    # gyro bias
    gxs, gys, gzs = [], [], []

    last_ax, last_ay, last_az = read_accel_raw()
    last_accmag = mag3(last_ax, last_ay, last_az)

    while time() - t0 < CALIB_S:
        ax, ay, az = read_accel_raw()
        gx, gy, gz = read_gyro_raw()

        gxs.append(gx); gys.append(gy); gzs.append(gz)

        accmag = mag3(ax, ay, az)
        accmag_samples.append(accmag)

        dax, day, daz = ax-last_ax, ay-last_ay, az-last_az
        jerk = mag3(dax, day, daz)
        jerk_samples.append(jerk)

        accdelta = abs(accmag - last_accmag)
        accdelta_samples.append(accdelta)

        # gyro mag (raw during still)
        gyro_samples.append(mag3(gx, gy, gz))

        last_ax, last_ay, last_az = ax, ay, az
        last_accmag = accmag

        sleep(DT)

    bias_gx = stats.mean(gxs) if gxs else 0.0
    bias_gy = stats.mean(gys) if gys else 0.0
    bias_gz = stats.mean(gzs) if gzs else 0.0

    jerk_mu, jerk_sd = mean_std(jerk_samples)
    gyro_mu, gyro_sd = mean_std(gyro_samples)
    accdelta_mu, accdelta_sd = mean_std(accdelta_samples)
    accmag_mu, accmag_sd = mean_std(accmag_samples)

    jerk_th = max(MIN_JERK_TH, jerk_mu + K_JERK * jerk_sd)
    gyro_th = max(MIN_GYRO_TH, gyro_mu + K_GYRO * gyro_sd)
    accdelta_th = max(MIN_ACCDELTA_TH, accdelta_mu + K_ACCDELTA * accdelta_sd)

    print("[CALIB] done.")
    print(f"  gyro_bias=({bias_gx:.1f},{bias_gy:.1f},{bias_gz:.1f})")
    print(f"  jerk:     mean={jerk_mu:.1f} sd={jerk_sd:.1f}  -> TH={jerk_th:.1f}")
    print(f"  gyro_mag: mean={gyro_mu:.1f} sd={gyro_sd:.1f}  -> TH={gyro_th:.1f}")
    print(f"  accΔmag:  mean={accdelta_mu:.1f} sd={accdelta_sd:.1f} -> TH={accdelta_th:.1f}")
    print(f"  acc_mag baseline≈{accmag_mu:.1f} (sd={accmag_sd:.1f})\n")

    print("Columns: t | acc_mag | jerk | gyro_mag | accΔ | flags")
    print("--------------------------------------------------------------")

    last_ax, last_ay, last_az = read_accel_raw()
    last_accmag = mag3(last_ax, last_ay, last_az)
    last_crash_t = 0.0

    while True:
        try:
            ax, ay, az = read_accel_raw()
            gx, gy, gz = read_gyro_raw()

            # remove gyro bias
            gx2 = gx - bias_gx
            gy2 = gy - bias_gy
            gz2 = gz - bias_gz

            accmag = mag3(ax, ay, az)
            gyro_mag = mag3(gx2, gy2, gz2)

            dax, day, daz = ax-last_ax, ay-last_ay, az-last_az
            jerk = mag3(dax, day, daz)

            accdelta = abs(accmag - last_accmag)

            flags = []
            if jerk > jerk_th * 0.35 or gyro_mag > gyro_th * 0.35:
                flags.append("MOVE")

            crash = (jerk >= jerk_th) or (gyro_mag >= gyro_th) or (accdelta >= accdelta_th)
            if crash and (time() - last_crash_t) >= COOLDOWN_S:
                flags.append("CRASH!")
                last_crash_t = time()

            ts = time()
            print(f"{ts:10.3f} | {accmag:7.1f} | {jerk:6.1f} | {gyro_mag:7.1f} | {accdelta:6.1f} | {' '.join(flags) if flags else '-'}")

            last_ax, last_ay, last_az = ax, ay, az
            last_accmag = accmag

        except Exception as e:
            print(f"{time():10.3f} | ERROR: {e}")

        sleep(DT)

if __name__ == "__main__":
    main()
