#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

try:
    import serial
except Exception:
    serial = None

# import class của bạn (đổi tên file cho đúng)
from perception_planner import PerceptionPlanner


def main():
    # ====== chỉnh 2 cái này theo robot bạn ======
    serial_port = "/dev/ttyUSB0"
    baud = 115200

    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        serial_port=serial_port,
        baud=baud,
        safe_dist_cm=50.0,
        enable_imu=False,
    )

    # chạy camera + uart loop (để class vẫn update decision/mini-map nếu bạn cần)
    planner.start()

    if serial is None:
        print("❌ pyserial chưa cài. Chạy: pip3 install pyserial")
        return

    # mở UART riêng để lấy temp + dist (vì class hiện chỉ lưu dist)
    try:
        ser = serial.Serial(serial_port, baud, timeout=1)
        time.sleep(1.5)
    except Exception as e:
        print(f"❌ Không mở được UART {serial_port}: {e}")
        planner.stop()
        return

    print("✅ Logging temp + distance... (Ctrl+C để dừng)")
    print("Format mong đợi từ ESP32: timestamp,temp,humidity,ultrasonic_cm")
    print("-" * 60)

    last_print = 0.0
    temp_c = None
    dist_cm = None
    humid = None

    try:
        while True:
            raw = ser.readline()
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) >= 4:
                # ví dụ: 1700000000,28.6,62.1,45.2
                try:
                    temp_c = float(parts[1])
                    humid = float(parts[2])
                    dist_cm = float(parts[3])
                except Exception:
                    continue

            # in log mỗi 0.5s
            now = time.time()
            if now - last_print >= 0.5:
                st = planner.get_state()
                # dist từ class (uart_dist_cm) và dist đọc trực tiếp (dist_cm)
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"T={temp_c if temp_c is not None else 'NA'}°C  "
                    f"H={humid if humid is not None else 'NA'}%  "
                    f"D={dist_cm if dist_cm is not None else 'NA'}cm  "
                    f"(planner.uart_dist_cm={st.uart_dist_cm})  "
                    f"decision={st.decision}  reason={st.reason}"
                )
                last_print = now

    except KeyboardInterrupt:
        print("\n Stopped by user.")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        planner.stop()


if __name__ == "__main__":
    main()
