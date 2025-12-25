#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import math
import threading
import subprocess
from typing import List, Tuple

from flask import Flask, Response, jsonify, request
from PIL import Image, ImageDraw

# =======================
# CONFIG
# =======================
# Path tới ultra_simple đã build từ rplidar_sdk
# Ví dụ của bạn: ~/rplidar_sdk/output/Linux/Release/ultra_simple
RPLIDAR_BIN = os.environ.get(
    "RPLIDAR_BIN",
    os.path.expanduser("~/rplidar_sdk/output/Linux/Release/ultra_simple")
)

RPLIDAR_PORT = os.environ.get("RPLIDAR_PORT", "/dev/ttyUSB0")
RPLIDAR_BAUD = int(os.environ.get("RPLIDAR_BAUD", "460800"))  # C1 thường là 460800

HTTP_HOST = os.environ.get("HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("PORT", "9399"))

# Map render config
IMG_SIZE = int(os.environ.get("IMG_SIZE", "700"))   # px
MAX_RANGE_M = float(os.environ.get("MAX_RANGE_M", "12.0"))  # C1 12m
POINT_LIMIT = int(os.environ.get("POINT_LIMIT", "2000"))    # giữ tối đa N điểm mới nhất

# scale: bán kính ảnh tương ứng MAX_RANGE_M
CENTER = IMG_SIZE // 2
RADIUS = int(IMG_SIZE * 0.47)
SCALE = RADIUS / MAX_RANGE_M  # px per meter

# =======================
# GLOBAL STATE
# =======================
app = Flask(__name__)

lock = threading.Lock()
latest_points_xy: List[Tuple[float, float, int]] = []  # (x_m, y_m, quality)
latest_ts = 0.0
status = {
    "running": False,
    "last_error": "",
    "port": RPLIDAR_PORT,
    "baud": RPLIDAR_BAUD,
    "bin": RPLIDAR_BIN,
}

# ultra_simple output sample:
# theta: 353.17 Dist: 02277.00 Q: 47
LINE_RE = re.compile(r"theta:\s*([0-9.]+)\s+Dist:\s*([0-9.]+)\s+Q:\s*(\d+)")

proc = None
stop_flag = False


def polar_to_xy_m(theta_deg: float, dist_mm: float) -> Tuple[float, float]:
    # dist in mm -> meters
    d_m = dist_mm / 1000.0
    th = math.radians(theta_deg)
    # coordinate: x forward/right? Tuỳ bạn, ở đây:
    # x = d*cos, y = d*sin (mặt phẳng)
    x = d_m * math.cos(th)
    y = d_m * math.sin(th)
    return x, y


def start_ultra_simple():
    """
    Spawn ultra_simple and parse stdout to points.
    """
    global proc, stop_flag, latest_ts

    stop_flag = False
    status["last_error"] = ""

    # IMPORTANT: dùng đúng cú pháp của SDK
    # ./ultra_simple --channel --serial /dev/ttyUSB0 460800
    cmd = [
        "stdbuf", "-oL", "-eL",  # line-buffer stdout/stderr
        RPLIDAR_BIN,
        "--channel", "--serial",
        RPLIDAR_PORT,
        str(RPLIDAR_BAUD),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as e:
        status["running"] = False
        status["last_error"] = f"Cannot start ultra_simple: {e}"
        return

    status["running"] = True

    # read loop
    try:
        for line in proc.stdout:
            if stop_flag:
                break

            m = LINE_RE.search(line)
            if not m:
                # bạn có thể bật debug nếu muốn xem log:
                # print(line.strip())
                continue

            theta = float(m.group(1))
            dist_mm = float(m.group(2))
            q = int(m.group(3))

            # bỏ điểm quá xa / dist=0
            if dist_mm <= 1:
                continue
            if dist_mm / 1000.0 > MAX_RANGE_M:
                continue

            x_m, y_m = polar_to_xy_m(theta, dist_mm)

            with lock:
                latest_points_xy.append((x_m, y_m, q))
                if len(latest_points_xy) > POINT_LIMIT:
                    latest_points_xy[:] = latest_points_xy[-POINT_LIMIT:]
                latest_ts = time.time()
    except Exception as e:
        status["last_error"] = f"Reader loop error: {e}"
    finally:
        status["running"] = False
        try:
            if proc and proc.poll() is None:
                proc.terminate()
        except:
            pass


def lidar_thread_main():
    """
    Auto-restart lidar if it exits.
    """
    while True:
        start_ultra_simple()

        # nếu bị stop thủ công thì dừng luôn
        if stop_flag:
            return

        # nếu crash => wait rồi restart
        time.sleep(1.0)


def render_minimap_png() -> bytes:
    """
    Draw points into a PNG image and return bytes.
    """
    with lock:
        pts = list(latest_points_xy)
        ts = latest_ts

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (10, 10, 12))
    d = ImageDraw.Draw(img)

    # draw border circle
    d.ellipse(
        (CENTER - RADIUS, CENTER - RADIUS, CENTER + RADIUS, CENTER + RADIUS),
        outline=(80, 80, 90),
        width=2,
    )
    # draw crosshair
    d.line((CENTER, 0, CENTER, IMG_SIZE), fill=(35, 35, 40), width=1)
    d.line((0, CENTER, IMG_SIZE, CENTER), fill=(35, 35, 40), width=1)

    # draw points
    # màu theo quality (đơn giản)
    for (x_m, y_m, q) in pts:
        px = int(CENTER + x_m * SCALE)
        py = int(CENTER - y_m * SCALE)  # invert y for screen
        if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
            # quality 0..255 => brightness
            br = max(60, min(255, q * 4))
            d.point((px, py), fill=(br, br, br))

    # draw center
    d.ellipse((CENTER - 3, CENTER - 3, CENTER + 3, CENTER + 3), fill=(255, 80, 80))

    # draw status text
    age = time.time() - ts if ts else 999
    txt = f"port={RPLIDAR_PORT} baud={RPLIDAR_BAUD} pts={len(pts)} age={age:.1f}s"
    d.text((10, 10), txt, fill=(220, 220, 220))

    # encode png
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =======================
# FLASK ROUTES
# =======================
@app.get("/")
def home():
    # trang web đơn giản auto refresh ảnh
    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>RPLIDAR C1 MiniMap</title>
      <style>
        body {{ background:#0b0b0f; color:#ddd; font-family: Arial; }}
        .wrap {{ display:flex; gap:20px; align-items:flex-start; padding:20px; }}
        img {{ border:1px solid #333; border-radius:12px; }}
        code {{ color:#9ad; }}
        button {{ padding:8px 12px; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div>
          <h2>RPLIDAR C1 Minimap</h2>
          <img id="mm" width="{IMG_SIZE}" height="{IMG_SIZE}" src="/minimap.png?t={int(time.time())}"/>
          <div style="margin-top:10px;">
            <button onclick="reloadImg()">Reload</button>
            <button onclick="fetch('/api/restart', {{method:'POST'}}).then(()=>alert('Restart requested'));">Restart Lidar</button>
          </div>
        </div>
        <div>
          <h3>Status</h3>
          <pre id="st"></pre>
          <p>Endpoints:</p>
          <ul>
            <li><code>/minimap.png</code></li>
            <li><code>/api/status</code></li>
            <li><code>/api/points</code></li>
          </ul>
        </div>
      </div>
      <script>
        function reloadImg(){{
          document.getElementById("mm").src = "/minimap.png?t=" + Date.now();
        }}
        async function tick(){{
          reloadImg();
          const r = await fetch("/api/status");
          const j = await r.json();
          document.getElementById("st").textContent = JSON.stringify(j, null, 2);
        }}
        setInterval(tick, 800);
        tick();
      </script>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.get("/minimap.png")
def minimap_png():
    png = render_minimap_png()
    return Response(png, mimetype="image/png")


@app.get("/api/status")
def api_status():
    with lock:
        pts = len(latest_points_xy)
        ts = latest_ts
    return jsonify({
        "running": status["running"],
        "port": status["port"],
        "baud": status["baud"],
        "bin": status["bin"],
        "points_buffered": pts,
        "last_point_ts": ts,
        "age_s": (time.time() - ts) if ts else None,
        "last_error": status["last_error"],
        "pid": proc.pid if proc else None,
    })


@app.get("/api/points")
def api_points():
    # trả về points thô để bạn debug
    with lock:
        pts = latest_points_xy[-800:]  # giới hạn trả về
    return jsonify({
        "n": len(pts),
        "points": [{"x": x, "y": y, "q": q} for (x, y, q) in pts]
    })


@app.post("/api/restart")
def api_restart():
    # restart ultra_simple
    global stop_flag, proc
    stop_flag = True
    try:
        if proc and proc.poll() is None:
            proc.terminate()
    except:
        pass
    time.sleep(0.5)
    # start again in a new thread
    stop_flag = False
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    return jsonify({"ok": True})


def main():
    # start lidar reader thread
    threading.Thread(target=lidar_thread_main, daemon=True).start()
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False)


if __name__ == "__main__":
    main()
