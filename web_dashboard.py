#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Optional, Callable, Dict, Any
from flask import Flask, Response, request, jsonify

try:
    import numpy as np
except Exception:
    np = None


def _to_py_scalar(v):
    """Convert numpy scalar to python scalar."""
    if np is not None:
        try:
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
        except Exception:
            pass
    return v


def json_safe(obj: Any, _depth: int = 0) -> Any:
    """
    Recursively convert obj to JSON-serializable types.
    - numpy scalar -> python scalar
    - dict keys -> str
    - set/tuple -> list
    - unknown object -> str(obj)
    """
    if _depth > 12:
        return str(obj)

    obj = _to_py_scalar(obj)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[str(k)] = json_safe(v, _depth + 1)
        return out

    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v, _depth + 1) for v in obj]

    # bytes -> show length only (avoid huge json)
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": len(obj)}

    # fallback
    return str(obj)


class WebDashboard:
    """
    - /            HTML dashboard
    - /mjpeg       live video
    - /status      decision + sectors + reason + uart/mqtt + extras
    - /minimap.png mini-map
    - /cmd?move=FORWARD|BACK|TURN_LEFT|TURN_RIGHT|STOP
    - /health      quick health ping
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
        @self.app.get("/health")
        def health():
            return jsonify({"ok": True, "ts": time.time()})

        @self.app.get("/")
        def index():
            return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Matthew Robot Dashboard</title>
  <style>
    body { font-family: Arial; margin: 14px; background:#0b0b0b; color:#eee; }
    .row { display:flex; gap:14px; align-items:flex-start; }
    img { border:1px solid #444; border-radius:8px; background:#111; }
    button { padding:10px 14px; margin:4px; border-radius:10px; border:0; cursor:pointer; }
    pre { background:#111; color:#0f0; padding:10px; width:520px; border-radius:10px; overflow:auto; }
    .card { background:#131313; padding:10px; border-radius:12px; border:1px solid #2a2a2a; margin-bottom:10px; }
    .kv { display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; }
    .pill { background:#1b1b1b; border:1px solid #333; padding:6px 10px; border-radius:999px; }
    .err { color:#ff6b6b; }
    .ok { color:#7CFC90; }
  </style>
</head>
<body>
  <h2>Matthew Robot Dashboard</h2>

  <div class="row">
    <div>
      <div class="card">
        <div><b>Live Camera</b></div>
        <img id="cam" src="/mjpeg" width="640" height="480"/>
        <div class="err" id="camerr"></div>
      </div>
      <div class="card">
        <div><b>Mini-map</b></div>
        <img id="minimap" src="/minimap.png" width="220"/>
      </div>
    </div>

    <div style="width:580px;">
      <div class="card">
        <b>Quick Status</b>
        <div class="kv">
          <div class="pill">Decision: <span id="d">...</span></div>
          <div class="pill">Reason: <span id="r">...</span></div>
          <div class="pill">Dist(cm): <span id="dist">...</span></div>
          <div class="pill">Temp(°C): <span id="t">...</span></div>
          <div class="pill">Hum(%): <span id="h">...</span></div>
          <div class="pill">Manual: <span id="m">...</span></div>
          <div class="pill">MQTT: <span id="mq">...</span></div>
        </div>
        <div class="err" id="err"></div>
      </div>

      <div class="card">
        <div><b>Status JSON</b></div>
        <pre id="status">loading...</pre>
      </div>

      <div class="card">
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
  </div>

<script>
async function refresh(){
  const errEl = document.getElementById('err');
  try{
    const r = await fetch('/status', {cache:'no-store'});
    const j = await r.json();

    if(j && j.ok === false){
      errEl.textContent = 'SERVER ERROR: ' + (j.error || 'unknown');
      document.getElementById('status').textContent = JSON.stringify(j, null, 2);
    }else{
      errEl.textContent = '';
      document.getElementById('status').textContent = JSON.stringify(j, null, 2);

      document.getElementById('d').textContent = j.decision ?? 'NA';
      document.getElementById('r').textContent = j.reason ?? 'NA';
      document.getElementById('dist').textContent = (j.uart_dist_cm ?? j.uart_dist_raw_cm ?? 'NA');
      document.getElementById('t').textContent = (j.uart_temp_c ?? 'NA');
      document.getElementById('h').textContent = (j.uart_humid ?? 'NA');
      document.getElementById('m').textContent = (j.manual_active ? (j.manual_move ?? 'NA') : 'OFF');

      const mq = (j.mqtt_ok === true) ? ('OK dt=' + (j.mqtt_dt_ms ?? 'NA') + 'ms') : ('ERR');
      document.getElementById('mq').textContent = mq;

      document.getElementById('minimap').src = '/minimap.png?t=' + Date.now();
    }
  }catch(e){
    errEl.textContent = 'FETCH ERROR: ' + e;
  }
  setTimeout(refresh, 250);
}

async function cmd(m){
  await fetch('/cmd?move=' + m, {cache:'no-store'});
}

refresh();
</script>
</body>
</html>
"""

        @self.app.get("/mjpeg")
        def mjpeg():
            def gen():
                # fallback black jpeg (tiny) if camera not ready
                black = None
                try:
                    import cv2, numpy as np
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if ok:
                        black = buf.tobytes()
                except Exception:
                    black = None

                while True:
                    try:
                        frame = self.get_jpeg() if self.get_jpeg else None
                        if not frame:
                            frame = black
                        if frame:
                            yield (b"--frame\r\n"
                                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        else:
                            time.sleep(0.05)
                    except Exception:
                        time.sleep(0.1)

            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.get("/status")
        def status():
            try:
                data = self.get_status() if self.get_status else {}
                return jsonify(json_safe(data))
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)})

        @self.app.get("/minimap.png")
        def minimap():
            try:
                png = self.get_minimap_png() if self.get_minimap_png else b""
                return Response(png, mimetype="image/png")
            except Exception:
                return Response(b"", mimetype="image/png")

        @self.app.get("/cmd")
        def cmd():
            move = (request.args.get("move") or "STOP").upper()
            if self.on_manual_cmd:
                try:
                    self.on_manual_cmd(move)
                except Exception:
                    pass
            return jsonify({"ok": True, "move": move})

    def run(self):
        # threaded=True để /mjpeg + /status chạy cùng lúc mượt
        self.app.run(host=self.host, port=self.port, threaded=True)
