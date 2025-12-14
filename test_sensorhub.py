#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from perception_planner import PerceptionPlanner

def main():
    planner = PerceptionPlanner(
        cam_dev="/dev/video0",
        w=640, h=480, fps=30,
        sector_n=9,
        map_h=80,
        serial_port="/dev/ttyUSB0",   # đổi nếu ESP32 là /dev/ttyACM0
        baud=115200,
        safe_dist_cm=50.0,
        enable_imu=False,
    )

    planner.start()
    print("Logging temp + humid + distance (Ctrl+C để dừng)")
    print("UART format mong đợi: timestamp,temp,humidity,ultrasonic_cm")
    print("-" * 80)

    try:
        while True:
            st = planner.get_state()
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"T={st.uart_temp_c if st.uart_temp_c is not None else 'NA'}°C  "
                f"H={st.uart_humid if st.uart_humid is not None else 'NA'}%  "
                f"D={st.uart_dist_cm if st.uart_dist_cm is not None else 'NA'}cm  "
                f"decision={st.decision}  reason={st.reason}"
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        planner.stop()

if __name__ == "__main__":
    main()
