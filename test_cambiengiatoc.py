#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smbus2 import SMBus
from time import sleep, time
import math
import sys

# ====== I2C addresses (the ones you scanned) ======
ADDR_ACC = 0x36   # Acc + Gyro (SH3001)

BUS_ID = 1

# ====== regs (the ones from your code) ======
REG_AX = 0x01
REG_AY = 0x03
REG_AZ = 0x05
REG_GX = 0x07
REG_GY = 0x09
REG_GZ = 0x0B

# Optional WHO_AM_I (may differ by SH3001 implementation; safe read)
REG_WHOAMI_CANDIDATES = [0x30, 0x0F, 0x1A]  # try a few common ones
REG_TEMP_CANDIDATES = [0x0D, 0x0E, 0x09]    # your temp reg in sample was 0x09 (but that overlaps GY), so keep optional

# ====== Detection thresholds (tune if needed) ======
# raw units unknown -> use relative thresholding
JERK_THRESHOLD = 8000.0      # accel delta magnitude / sample
GYRO_SPIKE_THRESHOLD = 12000.0  # gyro magnitude (after bias removal)
CRASH_COOLDOWN_S = 1.0       # avoid spamming crash events

# ====== Calibration ======
CALIB_SECONDS = 2.5
SAMPLE_DT = 0.10  # 10Hz (match your sleep)

bus = SMBus(BUS_ID)

def read_u8(addr, reg):
    return bus.read_byte_data(addr, reg)

def read_word(addr, reg):
    hi = bus.read_byte_data(addr, reg)
    lo = bus.read_byte_data(addr, reg + 1)
    val = (hi << 8) | lo
    if val & 0x8000:  # signed
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

def try_read_whoami():
    for r in REG_WHOAMI_CANDIDATES:
        try:
            v = read_u8(ADDR_ACC, r)
            return r, v
        except Exception:
            pass
    return None, None

def try_read_temp():
    for r in REG_TEMP_CANDIDATES:
        try:
            raw = read_word(ADDR_ACC, r)
            # generic SH3001-ish conversion (may be wrong; use just for "changes")
            temp_c = raw / 512.0 + 23.0
            return r, raw, temp_c
        except Exception:
            pass
    return None, None, None

def vec_mag(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def now_str():
    t = time()
    ms = int((t - int(t)) * 1000)
    return f"{time():.3f}"

def main():
    print("=== IMU SH3001 QUICK TEST (PiDog) ===")
    print(f"I2C addr: 0x{ADDR_ACC:02X} bus: {BUS_ID}")

    # Basic read test
    try:
        ax, ay, az = read_accel_raw()
        gx, gy, gz = read_gyro_raw()
        print(f"[BOOT] raw read OK | ACC=({ax},{ay},{az}) GYRO=({gx},{gy},{gz})")
    except Exception as e:
        print("[BOOT] ERROR: cannot read IMU from I2C:", e)
        print("Tips: check wiring, i2c enabled, correct bus=1, address=0x36.")
        sys.exit(1)

    who_r, who_v = try_read_whoami()
    if who_r is not None:
        print(f"[BOOT] WHO_AM_I? reg=0x{who_r:02X} val=0x{who_v:02X} (optional)")
    else:
        print("[BOOT] WHO_AM_I read: not available (optional)")

    tr, traw, tc = try_read_temp()
    if tr is not None:
        print(f"[BOOT] TEMP? reg=0x{tr:02X} raw={traw} approx={tc:.2f}C (optional)")
    else:
        print("[BOOT] TEMP read: not available (optional)")

    # ====== Calibrate gyro bias ======
    print(f"\n[CALIB] Hold robot still for {CALIB_SECONDS:.1f}s... calibrating gyro bias")
    t0 = time()
    n = 0
    sum_gx = sum_gy = sum_gz = 0.0
    while time() - t0 < CALIB_SECONDS:
        gx, gy, gz = read_gyro_raw()
        sum_gx += gx
        sum_gy += gy
        sum_gz += gz
        n += 1
        sleep(SAMPLE_DT)

    bias_gx = sum_gx / max(1, n)
    bias_gy = sum_gy / max(1, n)
    bias_gz = sum_gz / max(1, n)
    print(f"[CALIB] done. gyro_bias=({bias_gx:.1f},{bias_gy:.1f},{bias_gz:.1f}) samples={n}\n")

    # ====== Main loop: detect movement + crash ======
    last_ax, last_ay, last_az = read_accel_raw()
    last_t = time()
    last_crash_t = 0.0

    print("Columns:")
    print("  t | acc_mag | jerk | gyro_mag | flags")
    print("-----------------------------------------------------------")

    while True:
        try:
            ax, ay, az = read_accel_raw()
            gx, gy, gz = read_gyro_raw()

            # remove gyro bias
            gx2 = gx - bias_gx
            gy2 = gy - bias_gy
            gz2 = gz - bias_gz

            acc_mag = vec_mag(ax, ay, az)
            gyro_mag = vec_mag(gx2, gy2, gz2)

            # jerk = delta accel magnitude-ish (raw)
            dax = ax - last_ax
            day = ay - last_ay
            daz = az - last_az
            jerk = vec_mag(dax, day, daz)

            flags = []
            moving = (jerk > (JERK_THRESHOLD * 0.25)) or (gyro_mag > (GYRO_SPIKE_THRESHOLD * 0.25))
            if moving:
                flags.append("MOVE")

            crash = (jerk >= JERK_THRESHOLD) or (gyro_mag >= GYRO_SPIKE_THRESHOLD)
            if crash and (time() - last_crash_t) >= CRASH_COOLDOWN_S:
                flags.append("CRASH!")
                last_crash_t = time()

            # print log
            tstamp = time()
            print(f"{tstamp:10.3f} | {acc_mag:8.1f} | {jerk:8.1f} | {gyro_mag:9.1f} | {' '.join(flags) if flags else '-'}")

            last_ax, last_ay, last_az = ax, ay, az
            last_t = tstamp

        except Exception as e:
            print(f"{time():10.3f} | ERROR: {e}")

        sleep(SAMPLE_DT)

if __name__ == "__main__":
    main()
