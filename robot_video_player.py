#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import signal
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 9900

# đổi đúng tên service mặt robot của bạn (ví dụ face3d.service)
FACE_SERVICE = os.environ.get("FACE_SERVICE", "face3d.service")

CHROMIUM = os.environ.get("CHROMIUM_BIN", "chromium-browser")

chromium_proc = None

def face_stop():
    # nếu mặt chạy bằng systemd user thì dùng --user
    subprocess.run(["systemctl", "stop", FACE_SERVICE], check=False)

def face_start():
    subprocess.run(["systemctl", "start", FACE_SERVICE], check=False)

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

def video_play(url: str):
    global chromium_proc
    if not url:
        return False

    # nhường màn hình cho video
    face_stop()
    video_stop()

    cmd = [
        CHROMIUM,
        "--autoplay-policy=no-user-gesture-required",
        "--kiosk",
        "--noerrdialogs",
        "--disable-infobars",
        "--app=" + url
    ]
    chromium_proc = subprocess.Popen(cmd)
    return True

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        b = json.dumps(payload).encode("utf-8")
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

        return self._json(404, {"ok": False, "error": "not found"})

def main():
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"robot_video_player listening on 127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    finally:
        video_stop()

if __name__ == "__main__":
    main()
