#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path

from robot_hat import Servo
from motion_controller import MotionController

POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"

# Tune these angles for your robot.
REAR_LIFT_ANGLES = {
    "P4": 80,   # rear hip left
    "P5": 30,   # rear knee left
    "P6": -70,  # rear hip right
    "P7": -30,  # rear knee right
}

FRONT_LIFT_ANGLES = {
    "P0": -20,  # front hip left
    "P1": 90,   # front knee left
    "P2": 20,   # front hip right
    "P3": -75,  # front knee right
}

HEAD_INIT_ANGLES = {
    "P8": 80,   # head yaw
    "P9": -70,  # head roll
    "P10": 90,  # head pitch
}


def clamp(angle: float) -> int:
    try:
        v = int(angle)
    except Exception:
        v = 0
    return max(-90, min(90, v))


def apply_angles(angles: dict[str, float], per_servo_delay: float = 0.03):
    for port, angle in angles.items():
        try:
            s = Servo(port)
            s.angle(clamp(angle))
        except Exception:
            pass
        time.sleep(per_servo_delay)


def smooth_pair(
    pA: str, a_start: int, a_end: int,
    pB: str, b_start: int, b_end: int,
    step: int = 1,
    delay: float = 0.03,
):
    sA = Servo(pA)
    sB = Servo(pB)

    a_start, a_end = clamp(a_start), clamp(a_end)
    b_start, b_end = clamp(b_start), clamp(b_end)

    a = a_start
    b = b_start

    try:
        sA.angle(a)
        sB.angle(b)
    except Exception:
        pass

    max_steps = max(abs(a_end - a_start), abs(b_end - b_start))
    if max_steps == 0:
        return

    step = max(1, int(abs(step)))

    for _ in range(max_steps):
        if a != a_end:
            a += step if a_end > a else -step
            if (a_end > a_start and a > a_end) or (a_end < a_start and a < a_end):
                a = a_end

        if b != b_end:
            b += step if b_end > b else -step
            if (b_end > b_start and b > b_end) or (b_end < b_start and b < b_end):
                b = b_end

        try:
            sA.angle(clamp(a))
            sB.angle(clamp(b))
        except Exception:
            pass

    time.sleep(delay)


def _try_create_rgb_device():
    try:
        import robot_hat
    except Exception as e:
        print(f"[LED] robot_hat import failed: {e}")
        return None

    led_num = int(os.environ.get("PIDOG_LED_NUM", "2"))
    led_pin = int(os.environ.get("PIDOG_LED_PIN", "12"))
    candidates = (
        "RGBStrip",
        "RGBStripWS2812",
        "RGBStripAPA102",
        "RGBLed",
        "RGBLED",
    )
    arg_sets = (
        (),
        (led_num,),
        (led_num, led_pin),
        (led_pin, led_num),
    )

    for cls_name in candidates:
        cls = getattr(robot_hat, cls_name, None)
        if not cls:
            continue
        for args in arg_sets:
            try:
                dev = cls(*args)
                print(f"[LED] init {cls_name} args={args}")
                return dev
            except Exception:
                continue

    return None


def set_led(motion: MotionController, color: str, bps: float = 0.5):
    dog = getattr(motion, "dog", None)
    if not dog:
        print("[LED] motion has no dog instance")
        return

    rs = getattr(dog, "rgb_strip", None)
    rl = getattr(dog, "rgb_led", None)
    if not rs and not rl:
        dev = _try_create_rgb_device()
        if dev:
            try:
                dog.rgb_strip = dev
                rs = dev
            except Exception:
                pass
        else:
            print("[LED] no rgb device available (rgb_strip init failed?)")
            return

    # 1) rgb_strip.set_mode("breath", color, bps=?)
    try:
        if rs:
            try:
                rs.set_mode("breath", color, bps=bps)
                return
            except TypeError:
                rs.set_mode("breath", color, bps)
                return
            except Exception:
                pass
    except Exception:
        pass

    # 2) rgb_strip.set_color / fill / show
    try:
        rs = getattr(dog, "rgb_strip", None)
        if rs:
            for fn in ("set_color", "fill"):
                if hasattr(rs, fn):
                    try:
                        getattr(rs, fn)(color)
                        if hasattr(rs, "show"):
                            rs.show()
                        return
                    except Exception:
                        pass
    except Exception:
        pass

    # 3) rgb_led.set_color / set_rgb
    try:
        rl = getattr(dog, "rgb_led", None)
        if rl:
            for fn in ("set_color", "set_rgb", "setColor"):
                if hasattr(rl, fn):
                    try:
                        getattr(rl, fn)(color)
                        return
                    except Exception:
                        pass
    except Exception:
        pass


def main():
    os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
    os.environ.setdefault("PULSE_SERVER", "")
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    os.environ.setdefault("PIDOG_SKIP_HEAD_INIT", "1")
    os.environ.setdefault("PIDOG_SKIP_MCU_RESET", "1")

    print("[TEST] set head init angles")
    apply_angles(HEAD_INIT_ANGLES, per_servo_delay=0.04)
    print("[TEST] head init done (hold position)")

    print("[TEST] lift rear legs (left then right)")
    smooth_pair("P4", 0, REAR_LIFT_ANGLES["P4"], "P5", 0, REAR_LIFT_ANGLES["P5"], step=1, delay=0.03)
    smooth_pair("P6", 0, REAR_LIFT_ANGLES["P6"], "P7", 0, REAR_LIFT_ANGLES["P7"], step=1, delay=0.03)
    print("[TEST] rear lift done (hold position)")
    time.sleep(2.0)

    print("[TEST] lift front legs")
    apply_angles(FRONT_LIFT_ANGLES, per_servo_delay=0.04)
    print("[TEST] front lift done (hold position)")
    time.sleep(2.0)

    print("[TEST] boot robot to stand")
    motion = MotionController(pose_file=POSE_FILE)
    motion.boot()
    print("[TEST] boot done")
    set_led(motion, "green", bps=0.4)
    print("[TEST] led green")

    print("[TEST] done")


if __name__ == "__main__":
    main()
