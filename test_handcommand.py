# test_handcommand.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

from handcommand import HandCommand, HandCmdCfg

# Nếu bạn muốn test support_stand thật: import boot helper của bạn
# from matthew_pidog_boot import MatthewPidogBootClass

def set_face_udp(emo: str):
    # demo: chỉ in
    print("[FACE]", emo)

def on_cmd_print(cmd: str):
    # demo: chỉ in
    print("[CMD ]", cmd)

def bark_print():
    print("[BARK] woof woof")

def main():
    cfg = HandCmdCfg(
        cam_dev="/dev/video0",
        cam_w=640,
        cam_h=480,
        cooldown_sec=0.9,
        draw_overlay=True,

        # ✅ file memory
        state_file="gesture_state.json",
        log_file="gesture_log.jsonl",

        # ✅ xóa bộ nhớ cũ khi start
        clear_old_memory_on_start=True,
    )

    # boot = MatthewPidogBootClass(pose_file="pidog_pose_config.txt")
    # (nếu muốn: boot.create() để init dog, nhưng test này không cần)

    hc = HandCommand(
        cfg=cfg,
        on_cmd=on_cmd_print,
        set_face=set_face_udp,
        on_bark=bark_print,
        boot_helper=None,  # hoặc boot nếu bạn muốn gọi support_stand thật
    )

    print("\n=== HandCommand TEST ===")
    print("Keys:")
    print("  e  : toggle enable/disable")
    print("  q  : quit")
    print("Start with ENABLED...\n")

    hc.start(enabled=True)

    # simple keyboard loop (không cần cv2.imshow)
    try:
        while True:
            s = input("(e=toggle, q=quit) > ").strip().lower()
            if s == "q":
                break
            if s == "e":
                new_state = not hc.is_enabled()
                hc.set_enabled(new_state)
                print("[TOGGLE] enabled =", new_state)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        hc.stop()
        print("Bye. State/log saved next to this file:")
        print(" -", Path(cfg.state_file).resolve())
        print(" -", Path(cfg.log_file).resolve())

if __name__ == "__main__":
    main()
