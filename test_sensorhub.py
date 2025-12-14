#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from perception_planner import PerceptionPlanner

def main():
    planner = PerceptionPlanner(
        serial_port="/dev/ttyUSB0",
        baud=115200,
        safe_dist_cm=10.0,        # chỉnh tuỳ bạn
        enable_camera=False,      # ✅ tắt camera để test UART cho chắc
        enable_imu=False,
        uart_debug=True,          # ✅ in raw UART giống code bạn test
        uart_print_every=0.2,
    )

    planner.start()
    print("✅ Test SensorHub: temp/hum/dist (Ctrl+C để dừng)")
    print("-" * 90)

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
        print("\n🛑 Stopped.")
    finally:
        planner.stop()

if __name__ == "__main__":
    main()
