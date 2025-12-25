import os
import time
from flask import Flask, Response, jsonify

from lidar_proc_reader import LidarProcReader
from map_renderer import render_minimap_png

PORT = int(os.environ.get("PORT", "9399"))
LIDAR_PORT = os.environ.get("LIDAR_PORT", "/dev/ttyUSB0")
LIDAR_BAUD = int(os.environ.get("LIDAR_BAUD", "460800"))
BRIDGE_PATH = os.environ.get("LIDAR_BRIDGE", "./lidar_bridge")

app = Flask(__name__)

lidar = LidarProcReader(bridge_path=BRIDGE_PATH, port=LIDAR_PORT, baud=LIDAR_BAUD, max_points=2500)
lidar.start()


@app.get("/health")
def health():
    pts, ts, err = lidar.get_points()
    age = round(time.time() - ts, 2) if ts else None
    return jsonify(ok=True, points=len(pts), last_ts=ts, age_s=age, err=err)


@app.get("/scan")
def scan():
    pts, ts, err = lidar.get_points()
    return jsonify(ok=True, last_ts=ts, err=err, points=pts[:1200])


@app.get("/minimap.png")
def minimap():
    pts, ts, err = lidar.get_points()
    png = render_minimap_png(pts, size=600, meters_range=6.0, rotate_deg=0.0, stamp=time.strftime("%H:%M:%S"))
    return Response(png, mimetype="image/png")


@app.get("/")
def home():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>RPLidar Minimap</title>
  <style>
    body{background:#111;color:#ddd;font-family:Arial;text-align:center}
    img{border:1px solid #333;border-radius:10px;margin-top:12px}
    .small{color:#888;font-size:12px}
  </style>
</head>
<body>
  <h2>RPLidar Minimap (C++ SDK bridge)</h2>
  <img id="m" src="/minimap.png" width="600" height="600"/>
  <div class="small">refresh 200ms</div>
  <script>
    const img=document.getElementById('m');
    setInterval(()=>{ img.src='/minimap.png?t='+Date.now(); },200);
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False)
    finally:
        lidar.stop()
