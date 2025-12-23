#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ==============================
# CONFIG (ENV)
# ==============================
HOST = os.environ.get("VIDEO_HOST", "127.0.0.1")
PORT = int(os.environ.get("VIDEO_PORT", "9900"))

# Face service name (systemd)
FACE_SERVICE = os.environ.get("FACE_SERVICE", "robot-face.service")
# "system" -> systemctl start/stop (needs root)
# "user"   -> systemctl --user start/stop (needs user unit)
FACE_SYSTEMD_MODE = os.environ.get("FACE_SYSTEMD_MODE", "system").strip().lower()

# Chromium binary (on your Pi: /usr/bin/chromium)
CHROMIUM = os.environ.get("CHROMIUM_BIN", "chromium")

# Kiosk flags
KIOSK = os.environ.get("KIOSK", "1").strip() == "1"

# Where Chromium stores profile (avoid first-run prompts)
CHROMIUM_USER_DATA_DIR = os.environ.get(
    "CHROMIUM_USER_DATA_DIR",
    "/tmp/robot_chromium_profile"
)

# Audio output controls
# 1) On Raspberry Pi: audio mode by amixer numid=3
#    auto / analog / hdmi
AUDIO_MODE = os.environ.get("AUDIO_MODE", "").strip().lower()  # "", "auto", "analog", "hdmi"
# 2) Force chromium ALSA device
#    ex: default, plughw:0,0, hw:0,0
ALSA_OUTPUT_DEVICE = os.environ.get("ALSA_OUTPUT_DEVICE", "").strip()

# Optional: disable face control (if you don't want systemctl at all)
DISABLE_FACE_CONTROL = os.environ.get("DISABLE_FACE_CONTROL", "0").strip() == "1"

# Extra Chromium flags if you want
EXTRA_CHROMIUM_FLAGS = os.environ.get("EXTRA_CHROMIUM_FLAGS", "").strip()

chromium_proc = None
current_url = None
is_playing = False


# ==============================
# HELPERS
# ==============================
def run_cmd(cmd, check=False, quiet=True):
    try:
        if quiet:
            return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=check)
        return subprocess.run(cmd, check=check)
    except Exception:
        return None


def systemctl(args):
    if DISABLE_FACE_CONTROL:
        return
    if FACE_SYSTEMD_MODE == "user":
        run_cmd(["systemctl", "--user"] + args, check=False, quiet=True)
    else:
        run_cmd(["systemctl"] + args, check=False, quiet=True)


def face_stop():
    if DISABLE_FACE_CONTROL:
        return
    systemctl(["stop", FACE_SERVICE])


def face_start():
    if DISABLE_FACE_CONTROL:
        return
    systemctl(["start", FACE_SERVICE])


def set_audio_mode_pi(mode: str):
    """
    Raspberry Pi audio select:
      amixer cset numid=3 0  -> auto
      amixer cset numid=3 1  -> analog (3.5mm jack)
      amixer cset numid=3 2  -> hdmi
    """
    if not mode:
        return

    mode = mode.lower().strip()
    val = None
    if mode == "auto":
        val = "0"
    elif mode == "analog":
        val = "1"
    elif mode == "hdmi":
        val = "2"
    else:
        return

    # Best-effort
    run_cmd(["amixer", "cset", "numid=3", val], check=False, quiet=True)


def normalize_youtube_url(url: str) -> str:
    """
    Convert youtu.be / watch?v= to embed autoplay URL (more reliable in kiosk).
    """
    u = (url or "").strip()
    if not u:
        return u

    # Already embed
    if "youtube.com/embed/" in u:
        # ensure autoplay
        if "autoplay=" not in u:
            join = "&" if "?" in u else "?"
            u = u + f"{join}autoplay=1&controls=1&fs=1&rel=0"
        return u

    # youtu.be/<id>?...
    if "youtu.be/" in u:
        vid = u.split("youtu.be/")[-1].split("?")[0].split("&")[0]
        if vid:
            return f"https://www.youtube.com/embed/{vid}?autoplay=1&controls=1&fs=1&rel=0"

    # youtube.com/watch?v=<id>
    if "youtube.com/watch" in u:
        qs = parse_qs(urlparse(u).query)
        vid = (qs.get("v") or [""])[0]
        if vid:
            return f"https://www.youtube.com/embed/{vid}?autoplay=1&controls=1&fs=1&rel=0"

    return u


def video_stop():
    global chromium_proc, current_url, is_playing
    if chromium_proc and chromium_proc.poll() is None:
        try:
            chromium_proc.terminate()
            chromium_proc.wait(timeout=2)
        except Exception:
            try:
                chromium_proc.kill()
            except Exception:
                pass
    chromium_proc = None
    current_url = None
    is_playing = False


def build_chromium_cmd(url: str):
    flags = [
        CHROMIUM,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-session-crashed-bubble",
        "--disable-infobars",
        "--noerrdialogs",
        "--autoplay-policy=no-user-gesture-required",
        "--disable-features=TranslateUI,MediaRouter",
        f"--user-data-dir={CHROMIUM_USER_DATA_DIR}",
    ]

    # Force ALSA output device if provided
    if ALSA_OUTPUT_DEVICE:
        flags.append(f"--alsa-output-device={ALSA_OUTPUT_DEVICE}")

    if KIOSK:
        flags += ["--kiosk"]

    # Use app mode so it looks clean
    flags.append("--app=" + url)

    if EXTRA_CHROMIUM_FLAGS:
        # split by spaces (simple)
        flags += EXTRA_CHROMIUM_FLAGS.split()

    return flags


def video_play(url: str) -> bool:
    global chromium_proc, current_url, is_playing
    if not url:
        return False

    url = normalize_youtube_url(url)

    # set audio route (Pi)
    set_audio_mode_pi(AUDIO_MODE)

    # show video => stop face & stop previous
    face_stop()
    video_stop()

    cmd = build_chromium_cmd(url)

    try:
        chromium_proc = subprocess.Popen(cmd)
        current_url = url
        is_playing = True
        return True
    except Exception as e:
        print("[VIDEO] failed to start chromium:", e)
        chromium_proc = None
        current_url = None
        is_playing = False
        return False


# ==============================
# HTTP SERVER
# ==============================
class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        b = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, fmt, *args):
        # reduce noisy logs
        return

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ok": True, "playing": is_playing, "url": current_url})
        if self.path == "/status":
            return self._json(200, {"ok": True, "playing": is_playing, "url": current_url})
        return self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            data = {}

        if self.path == "/play":
            url = (data.get("url") or "").strip()
            ok = video_play(url)
            return self._json(200, {"ok": ok, "action": "play", "url": current_url or url})

        if self.path == "/stop":
            video_stop()
            face_start()
            return self._json(200, {"ok": True, "action": "stop"})

        if self.path == "/face":
            video_stop()
            face_start()
            return self._json(200, {"ok": True, "action": "face"})

        return self._json(404, {"ok": False, "error": "not found"})


def shutdown_handler(signum, frame):
    try:
        video_stop()
    except Exception:
        pass
    sys.exit(0)


def main():
    os.makedirs(CHROMIUM_USER_DATA_DIR, exist_ok=True)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    server = HTTPServer((HOST, PORT), Handler)
    print(f"robot_video_player listening on {HOST}:{PORT}")
    print(f"FACE_SERVICE={FACE_SERVICE}  FACE_SYSTEMD_MODE={FACE_SYSTEMD_MODE}  DISABLE_FACE_CONTROL={DISABLE_FACE_CONTROL}")
    print(f"CHROMIUM={CHROMIUM}  KIOSK={KIOSK}  ALSA_OUTPUT_DEVICE={ALSA_OUTPUT_DEVICE or '(none)'}  AUDIO_MODE={AUDIO_MODE or '(none)'}")

    try:
        server.serve_forever()
    finally:
        video_stop()


if __name__ == "__main__":
    main()
