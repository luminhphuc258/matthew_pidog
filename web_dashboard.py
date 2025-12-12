#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Optional, Callable, Dict
from flask import Flask, Response, request, jsonify


class WebDashboard:
    """
    - /            HTML simple
    - /mjpeg       live video
    - /status      decision + sectors + reason + uart
    - /minimap.png mini-map
    - /cmd?move=FORWARD|BACK|TURN_LEFT|TURN_RIGHT|STOP
    """

    def __init__(
        self,
        host="0.0.0.0",
        port=8000,
        get_jpeg: Optional[Callable[[], Optional[bytes]]] = None,
        get_status: Optional[Callable[[], Dict]] = None,
        get_minimap_png: Optional[Callable[[], bytes]] = None,
        on_manual_cmd: Optional[Callable[[str], None]] = None,
    ):
        self.host, self.port = host, port
        self.get_jpeg = get_jpeg
        self.get_status = get_status
        self.get_minimap_png = get_minimap_png
        self.on_manual_cmd = on_manual_cmd

        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/")
        def index():
            return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Matthew Robot Dashboard</title>
  <style>
    body { font-family: Arial; margin: 14px; }
    .row { display:flex; gap:14px; align-items:flex-start; }
    img { border:1px solid #444; }
    button { padding:10px 14px; margin:4px; }
    pre { background:#111; color:#0f0; padding:10px; width:420px; }
  </style>
</head>
<body>
  <h2>Matthew Robot Dashboard</h2>
  <div class="row">
    <div>
      <div><b>Live Camera</b></div>
      <img src="/mjpeg" width="640" height="480"/>
      <div style="margin-top:10px"><b>Mini-map</b></div>
      <img id="minimap" src="/minimap.png" width="200"/>
    </div>
    <div>
      <div><b>Status</b></div>
      <pre id="status">loading...</pre>
      <div><b>Manual Control</b></div>
      <div>
        <button onclick="cmd('FORWARD')">FORWARD</button>
      </div>
      <div>
        <button onclick="cmd('TURN_LEFT')">LEFT</button>
        <button onclick="cmd('STOP')">STOP</button>
        <button onclick="cmd('TURN_RIGHT')">RIGHT</button>
      </div>
      <div>
        <button onclick="cmd('BACK')">BACK</button>
      </div>
    </div>
  </div>

<script>
async function refresh(){
  try{
    const r = await fetch('/status');
    const j = await r.json();
    document.getElementById('status').textContent = JSON.stringify(j, null, 2);
    document.getElementById('minimap').src = '/minimap.png?t=' + Date.now();
  }catch(e){}
  setTimeout(refresh, 300);
}
async function cmd(m){
  await fetch('/cmd?move=' + m);
}
refresh();
</script>
</body>
</html>
"""

        @self.app.get("/mjpeg")
        def mjpeg():
            def gen():
                while True:
                    frame = self.get_jpeg() if self.get_jpeg else None
                    if frame:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    time.sleep(0.03)
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.get("/status")
        def status():
            return jsonify(self.get_status() if self.get_status else {})

        @self.app.get("/minimap.png")
        def minimap():
            png = self.get_minimap_png() if self.get_minimap_png else b""
            return Response(png, mimetype="image/png")

        @self.app.get("/cmd")
        def cmd():
            move = (request.args.get("move") or "STOP").upper()
            if self.on_manual_cmd:
                self.on_manual_cmd(move)
            return jsonify({"ok": True, "move": move})

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True)
