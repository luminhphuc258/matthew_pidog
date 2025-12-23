#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
import signal
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("VIDEO_PORT", "9900"))

# Mặc định đúng theo service bạn đang có
FACE_SERVICE = os.environ.get("FACE_SERVICE", "robot-face.service")

# Nên set ENV CHROMIUM_BIN=/usr/bin/chromium trong .service
CHROMIUM = os.environ.get("CHROMIUM_BIN", "/usr/bin/chromium")

chromium_proc = None


def _run_systemctl(action: str, service: str):
    """
    Try user service first (systemctl --user), then system service.
    """
    # try --user
    try:
        subprocess.run(
            ["systemctl", "--user", action, service],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # không return luôn vì có trường hợp --user không có session -> fail silent
    except Exception:
        pass

    # fallback system service
    try:
        subprocess.run(
            ["sudo", "systemctl", action, service],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # fallback không sudo (nếu service cho phép)
        subprocess.run(
            ["systemctl", action, service],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def face_stop():
    _run_systemctl("stop", FACE_SERVICE)


def face_start():
    _run_systemctl("start", FACE_SERVICE)


def video_stop():
    global chromium_proc
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


VIDEO_USER = os.environ.get("VIDEO_USER", "matthewlupi")
CHROMIUM = os.environ.get("CHROMIUM_BIN", "/usr/bin/chromium")
DISPLAY = os.environ.get("DISPLAY", ":0")
XAUTH = os.environ.get("XAUTHORITY", f"/home/{VIDEO_USER}/.Xauthority")

def video_play(url: str):
    global chromium_proc
    if not url:
        return False

    # stop face (root sẽ làm được)
    face_stop()
    video_stop()

    # thêm autoplay param cho chắc
    if "youtube.com/watch" in url and "autoplay=" not in url:
        url = url + ("&" if "?" in url else "?") + "autoplay=1"

    cmd = [
        "runuser", "-l", VIDEO_USER, "-c",
        (
            f'DISPLAY={DISPLAY} XAUTHORITY={XAUTH} '
            f'{CHROMIUM} '
            f'--autoplay-policy=no-user-gesture-required '
            f'--kiosk --noerrdialogs --disable-infobars '
            f'--no-sandbox --disable-dev-shm-usage '
            f'--user-data-dir=/tmp/chromium-kiosk '
            f'--app="{url}"'
        )
    ]

    chromium_proc = subprocess.Popen(cmd)
    return True



class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # giảm spam log của BaseHTTPRequestHandler
        return

    def _json(self, code, payload):
        b = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

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
            return self._json(200, {"ok": ok, "action": "play", "url": url})

        if self.path == "/stop":
            video_stop()
            face_start()
            return self._json(200, {"ok": True, "action": "stop"})

        if self.path == "/face":
            video_stop()
            face_start()
            return self._json(200, {"ok": True, "action": "face"})

        if self.path == "/status":
            alive = bool(chromium_proc and chromium_proc.poll() is None)
            return self._json(200, {
                "ok": True,
                "playing": alive,
                "face_service": FACE_SERVICE,
                "chromium": CHROMIUM,
                "display": os.environ.get("DISPLAY", ""),
            })

        return self._json(404, {"ok": False, "error": "not found"})


def main():
    server = HTTPServer(("127.0.0.1", PORT), Handler)

    def _handle_term(sig, frame):
        try:
            server.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    print(f"robot_video_player listening on 127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    finally:
        video_stop()


if __name__ == "__main__":
    main()
