#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import argparse

import serial


def open_serial(port: str, baud: int):
    ser = serial.Serial(
        port,
        baudrate=baud,
        timeout=1,
        dsrdtr=False,
        rtscts=False,
    )
    # tránh reset board khi mở serial (nếu driver hỗ trợ)
    try:
        ser.setDTR(False)
        ser.setRTS(False)
    except Exception:
        pass

    time.sleep(0.4)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0", help="e.g. /dev/ttyUSB0 or /dev/n8r8")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--print-every", type=float, default=0.0, help="0=print every line, else throttle seconds")
    ap.add_argument("--log", default="", help="optional log file path")
    args = ap.parse_args()

    ser = None
    last_rx = 0.0
    last_print = 0.0

    logf = None
    if args.log:
        logf = open(args.log, "a", encoding="utf-8")
        logf.write("# ts_local, raw_line\n")
        logf.flush()

    print("=== UART TEST N8R8 -> PI (Ctrl+C to stop) ===")
    print("PORT:", args.port, "BAUD:", args.baud)
    print("Expect CSV: timestamp,temp,humid,ultrasonic_cm")
    print("-" * 80)

    try:
        while True:
            if ser is None:
                try:
                    ser = open_serial(args.port, args.baud)
                    print(f"[OK] opened {args.port} @ {args.baud}")
                    last_rx = time.time()
                except Exception as e:
                    print(f"[ERR] open failed: {e}")
                    time.sleep(1.0)
                    continue

            try:
                raw = ser.readline()
                if not raw:
                    # watchdog: nếu >3s không nhận gì thì báo
                    if time.time() - last_rx > 3.0:
                        print(f"[WARN] no data for {time.time() - last_rx:.1f}s")
                        last_rx = time.time()  # tránh spam warn
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                now = time.time()
                last_rx = now

                # log raw nếu muốn
                if logf:
                    logf.write(f"{now:.3f},{line}\n")
                    logf.flush()

                # throttle print nếu cần
                if args.print_every > 0:
                    if now - last_print < args.print_every:
                        continue
                    last_print = now

                # in raw
                print("UART >>", line)

                # parse thử CSV
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    try:
                        ts_ms = int(float(parts[0]))
                        t = float(parts[1])
                        h = float(parts[2])
                        d = float(parts[3])
                        print(f"PARSE  : ts={ts_ms}  temp={t:.2f}C  hum={h:.2f}%  dist={d:.2f}cm")
                    except Exception:
                        print("PARSE  : (not numeric CSV)")
                else:
                    print("PARSE  : (not CSV-4 fields)")

            except (serial.SerialException, OSError) as e:
                # đây là case USB disconnect / reconnect
                print(f"[UART] disconnected: {e}")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(0.8)

    except KeyboardInterrupt:
        print("\n[EXIT] stopped.")
    finally:
        try:
            if ser:
                ser.close()
        except Exception:
            pass
        if logf:
            logf.close()


if __name__ == "__main__":
    main()
