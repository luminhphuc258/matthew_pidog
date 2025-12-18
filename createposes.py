#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import curses
from pathlib import Path

# robot_hat Servo (SunFounder)
from robot_hat import Servo

CONFIG_FILE = "mypose_config.txt"  # JSON: {"P0": 0, ..., "P11": 0}
SERVO_PORTS = [f"P{i}" for i in range(12)]  # P0..P11 (8 legs + 3 head + 1 tail)

ANGLE_MIN = -90
ANGLE_MAX = 90

DEFAULT_STEP = 1
DELAY_BETWEEN_WRITES = 0.01  # seconds

# ==== TƯ THẾ CHUẨN: ROBOT NGỒI + ĐẦU NHÌN THẲNG ====
# Lấy từ file ~/pidog/examples/pidog_pose_config.txt trong hình
BASE_POSE_CONFIG = {
    "P0": 0,
    "P1": 18,
    "P2": 4,
    "P3": 0,
    "P4": 8,
    "P5": 17,
    "P6": 3,
    "P7": 4,
    "P8": -32,   # head yaw
    "P9": 90,    # head roll
    "P10": -90,  # head pitch
    "P11": 0,    # tail
}


def clamp(v, lo=ANGLE_MIN, hi=ANGLE_MAX):
    try:
        v = int(v)
    except Exception:
        v = 0
    return max(lo, min(hi, v))


def default_config():
    """
    Config mặc định = tư thế chuẩn (ngồi + đầu thẳng).
    Những servo không có trong BASE_POSE_CONFIG thì = 0.
    """
    cfg = {port: 0 for port in SERVO_PORTS}
    for k, v in BASE_POSE_CONFIG.items():
        if k in cfg:
            cfg[k] = clamp(v)
    return cfg


def load_config(path: Path):
    """
    Nếu file chưa tồn tại hoặc bị lỗi => trả về default_config()
    (tư thế chuẩn).
    """
    if not path.exists():
        return default_config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = default_config()
        if isinstance(data, dict):
            for k, v in data.items():
                if k in cfg:
                    cfg[k] = clamp(v)
        return cfg
    except Exception:
        # file hỏng => reset
        return default_config()


def save_config(path: Path, cfg: dict):
    # chỉ lưu đúng key P0..P11
    out = {p: int(clamp(cfg.get(p, 0))) for p in SERVO_PORTS}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def servo_set_angle(servo_obj, angle: int):
    """
    robot_hat Servo API có thể khác version.
    Mình thử vài method phổ biến để chắc chắn chạy được.
    """
    angle = clamp(angle)
    for method_name in ("angle", "write", "set_angle", "setAngle"):
        m = getattr(servo_obj, method_name, None)
        if callable(m):
            m(angle)
            return
    # fallback: thử call object (ít gặp)
    if callable(servo_obj):
        servo_obj(angle)
        return
    raise RuntimeError("Servo object has no known angle/write method")


def apply_all(servos: dict, cfg: dict):
    for port in SERVO_PORTS:
        servo_set_angle(servos[port], cfg.get(port, 0))
        time.sleep(DELAY_BETWEEN_WRITES)


def apply_one(servos: dict, port: str, angle: int):
    servo_set_angle(servos[port], angle)
    time.sleep(DELAY_BETWEEN_WRITES)


HELP_LINES = [
    "Controls:",
    "  ↑/↓ : chọn servo (P0..P11)",
    "  ←/→ : giảm/tăng góc theo STEP",
    "  1/2/5/0 : STEP = 1 / 2 / 5 / 10",
    "  r : reset servo hiện tại = 0",
    "  R : reset TẤT CẢ servo = 0",
    "  a : apply (đẩy config xuống servo ngay)",
    "  s : save file config",
    "  l : load file config",
    "  q : quit (tự động lưu)",
]


def port_label(i: int) -> str:
    if 0 <= i <= 7:
        return f"P{i} (leg {i+1})"
    if i == 8:
        return "P8 (head yaw)"
    if i == 9:
        return "P9 (head roll)"
    if i == 10:
        return "P10 (head pitch)"
    if i == 11:
        return "P11 (tail)"
    return f"P{i}"


def draw(stdscr, cfg: dict, idx: int, step: int, status: str, config_path: Path):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    title = f"Matthew PiDog Servo Editor  |  file: {config_path.name}"
    stdscr.addstr(0, 0, title[:w-1], curses.A_BOLD)

    stdscr.addstr(
        1, 0,
        f"Running as UID={os.geteuid()}  (Tip: chạy sudo để điều khiển servo)",
        curses.A_DIM,
    )
    stdscr.addstr(
        2, 0,
        f"Current: {port_label(idx)}   angle={cfg[SERVO_PORTS[idx]]:+d}   STEP={step}",
        curses.A_BOLD,
    )

    # list ports
    start_y = 4
    for i, port in enumerate(SERVO_PORTS):
        y = start_y + i
        if y >= h - 9:
            break
        val = cfg.get(port, 0)
        line = f"{port:>3} : {val:+4d}   {port_label(i)}"
        if i == idx:
            stdscr.addstr(y, 0, line[:w-1], curses.A_REVERSE | curses.A_BOLD)
        else:
            stdscr.addstr(y, 0, line[:w-1])

    # help
    help_y = h - 8
    if help_y < start_y + len(SERVO_PORTS) + 1:
        help_y = start_y + len(SERVO_PORTS) + 1
    if help_y < h - 1:
        stdscr.hline(help_y - 1, 0, ord("-"), max(0, w-1))
        for i, line in enumerate(HELP_LINES):
            y = help_y + i
            if y >= h - 1:
                break
            stdscr.addstr(y, 0, line[:w-1], curses.A_DIM)

    # status
    if status:
        stdscr.addstr(h - 1, 0, status[:w-1], curses.A_BOLD)

    stdscr.refresh()


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    # file path (cùng folder với script)
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / CONFIG_FILE

    # init servos
    servos = {port: Servo(port) for port in SERVO_PORTS}

    # ==== STARTUP: LOAD + APPLY CONFIG NGAY ====
    if config_path.exists():
        # Có file -> load + apply
        cfg = load_config(config_path)
        try:
            apply_all(servos, cfg)
            status = f"Đã load + apply config từ {config_path.name}. Bắt đầu chỉnh servo."
        except Exception as e:
            status = f"⚠️ Load OK nhưng apply ALL thất bại: {e} (try sudo?)"
    else:
        # Không có file -> dùng tư thế chuẩn, lưu file + apply
        cfg = default_config()
        try:
            save_config(config_path, cfg)
        except Exception as e:
            # lưu lỗi thì vẫn tiếp tục, vì cfg vẫn dùng được
            status = f"⚠️ Không lưu được file mới: {e}"
        try:
            apply_all(servos, cfg)
            status = (
                f"Không thấy {config_path.name}, đã tạo với tư thế chuẩn và apply xuống servo. "
                "Bắt đầu chỉnh servo."
            )
        except Exception as e:
            status = f"⚠️ Tạo file mới nhưng apply ALL thất bại: {e} (try sudo?)"

    idx = 0
    step = DEFAULT_STEP

    draw(stdscr, cfg, idx, step, status, config_path)

    while True:
        key = stdscr.getch()

        if key in (ord("q"), ord("Q")):
            # auto-save trước khi thoát
            try:
                save_config(config_path, cfg)
                status = f"Quit. Đã auto-save -> {config_path}"
            except Exception as e:
                status = f"⚠️ Quit nhưng SAVE thất bại: {e}"
            draw(stdscr, cfg, idx, step, status, config_path)
            break

        elif key == curses.KEY_UP:
            idx = (idx - 1) % len(SERVO_PORTS)

        elif key == curses.KEY_DOWN:
            idx = (idx + 1) % len(SERVO_PORTS)

        elif key == curses.KEY_LEFT:
            port = SERVO_PORTS[idx]
            cfg[port] = clamp(cfg[port] - step)
            try:
                apply_one(servos, port, cfg[port])
                status = f"Applied {port} = {cfg[port]:+d}"
            except Exception as e:
                status = f"⚠️ Apply failed: {e} (try sudo?)"

        elif key == curses.KEY_RIGHT:
            port = SERVO_PORTS[idx]
            cfg[port] = clamp(cfg[port] + step)
            try:
                apply_one(servos, port, cfg[port])
                status = f"Applied {port} = {cfg[port]:+d}"
            except Exception as e:
                status = f"⚠️ Apply failed: {e} (try sudo?)"

        elif key in (ord("1"),):
            step = 1
            status = "STEP = 1"
        elif key in (ord("2"),):
            step = 2
            status = "STEP = 2"
        elif key in (ord("5"),):
            step = 5
            status = "STEP = 5"
        elif key in (ord("0"),):
            step = 10
            status = "STEP = 10"

        elif key == ord("r"):
            port = SERVO_PORTS[idx]
            cfg[port] = 0
            try:
                apply_one(servos, port, 0)
                status = f"Reset {port} = 0"
            except Exception as e:
                status = f"⚠️ Reset failed: {e} (try sudo?)"

        elif key == ord("R"):
            for p in SERVO_PORTS:
                cfg[p] = 0
            try:
                apply_all(servos, cfg)
                status = "Reset ALL servos = 0 (applied)"
            except Exception as e:
                status = f"⚠️ Reset ALL failed: {e} (try sudo?)"

        elif key in (ord("a"), ord("A")):
            try:
                apply_all(servos, cfg)
            except Exception as e:
                status = f"⚠️ Apply ALL failed: {e} (try sudo?)"
            else:
                status = "Applied ALL servos từ config hiện tại."

        elif key in (ord("s"), ord("S")):
            try:
                save_config(config_path, cfg)
                status = f"Saved -> {config_path}"
            except Exception as e:
                status = f"⚠️ Save failed: {e}"

        elif key in (ord("l"), ord("L")):
            cfg = load_config(config_path)
            status = f"Loaded <- {config_path} (nhấn 'a' nếu muốn apply lại toàn bộ)"

        else:
            status = f"(ignored key: {key})"

        draw(stdscr, cfg, idx, step, status, config_path)


if __name__ == "__main__":
    curses.wrapper(main)
