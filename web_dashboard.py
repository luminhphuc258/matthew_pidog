#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict

from flask import Flask, Response, request, jsonify

# MQTT
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None


@dataclass
class MqttSensorState:
    ok: bool = False
    last_ts: float = 0.0
    dt_ms: Optional[int] = None
    dist_cm: Optional[float] = None
    temp_c: Optional[float] = None
    humid: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None
    err: Optional[str] = None


def _pick(d: Dict[str, Any], keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


class WebDashboard:
    """
    - /            HTML dashboard (camera + quick status + toggles)
    - /mjpeg       live video
    - /status      mqtt sensor state + toggles
    - /cmd         manual command (optional)
    - /toggle_listen  toggle active listening
    - /toggle_auto    toggle auto-move
    - /health      quick ping
    """

    def __init__(
        self,
        host="0.0.0.0",
        port=8000,
        get_jpeg: Optional[Callable[[], Optional[bytes]]] = None,

        # optional manual command callback (FORWARD/BACK/LEFT/RIGHT/STOP)
        on_manual_cmd: Optional[Callable[[str], None]] = None,

        # MQTT config (subscribe sensors directly here)
        mqtt_enable: bool = True,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_user: Optional[str] = None,
        mqtt_pass: Optional[str] = None,
        mqtt_topic: str = "/pidog/sensorhubdata",
        mqtt_client_id: str = "pidog-webdash",
        mqtt_tls: bool = False,
        mqtt_insecure: bool = True,
        mqtt_debug: bool = False,
    ):
        self.host, self.port = host, port
        self.get_jpeg = get_jpeg
        self.on_manual_cmd = on_manual_cmd

        self.mqtt_enable = mqtt_enable
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.mqtt_user = mqtt_user
        self.mqtt_pass = mqtt_pass
        self.mqtt_topic = mqtt_topic
        self.mqtt_client_id = mqtt_client_id
        self.mqtt_tls = mqtt_tls
        self.mqtt_insecure = mqtt_insecure
        self.mqtt_debug = mqtt_debug

        self._lock = threading.Lock()
        self._sensor = MqttSensorState()

        # toggles
        self._listen_on = False
        self._auto_on = False

        self.app = Flask(__name__)
        self._setup_routes()

        # start mqtt background
        self._mqtt_client = None
        if self.mqtt_enable:
            self._start_mqtt()

    # ---------- Public getters for main loop ----------
    def is_listen_on(self) -> bool:
        with self._lock:
            return bool(self._listen_on)

    def is_auto_on(self) -> bool:
        with self._lock:
            return bool(self._auto_on)

    # ---------- MQTT ----------
    def _start_mqtt(self):
        if mqtt is None:
            with self._lock:
                self._sensor.ok = False
                self._sensor.err = "paho-mqtt not installed"
            return

        c = mqtt.Client(client_id=self.mqtt_client_id, clean_session=True)

        if self.mqtt_user:
            c.username_pw_set(self.mqtt_user, self.mqtt_pass or "")

        if self.mqtt_tls:
            # basic TLS (works with EMQX)
            c.tls_set()
            if self.mqtt_insecure:
                c.tls_insecure_set(True)

        def on_connect(client, userdata, flags, rc):
            if self.mqtt_debug:
                print("[WebDash MQTT] connect rc=", rc)
            if rc == 0:
                client.subscribe(self.mqtt_topic, qos=0)
                with self._lock:
                    self._sensor.ok = True
                    self._sensor.err = None
            else:
                with self._lock:
                    self._sensor.ok = False
                    self._sensor.err = f"connect rc={rc}"

        def on_disconnect(client, userdata, rc):
            if self.mqtt_debug:
                print("[WebDash MQTT] disconnect rc=", rc)
            with self._lock:
                self._sensor.ok = False
                self._sensor.err = f"disconnect rc={rc}"

        def on_message(client, userdata, msg):
            now = time.time()
            payload = msg.payload
            data = None
            err = None

            # decode
            try:
                s = payload.decode("utf-8", errors="ignore").strip()
                data = json.loads(s)
                if not isinstance(data, dict):
                    data = {"value": data}
            except Exception as e:
                err = f"parse_json: {e}"
                data = None

            with self._lock:
                # dt
                if self._sensor.last_ts > 0:
                    self._sensor.dt_ms = int((now - self._sensor.last_ts) * 1000)
                self._sensor.last_ts = now

                if data is not None:
                    # try multiple possible key names
                    dist = _pick(data, ["uart_dist_cm", "dist_cm", "distance_cm", "distance", "dist", "range_cm"])
                    temp = _pick(data, ["uart_temp_c", "temp_c", "temperature_c", "temperature", "temp", "t"])
                    hum  = _pick(data, ["uart_humid", "humid", "humidity", "hum", "h"])

                    self._sensor.dist_cm = _to_float(dist)
                    self._sensor.temp_c  = _to_float(temp)
                    self._sensor.humid   = _to_float(hum)
                    self._sensor.raw = data
                    self._sensor.err = None
                    self._sensor.ok = True
                else:
                    # keep last values, just note error
                    self._sensor.err = err
                    self._sensor.ok = False

        c.on_connect = on_connect
        c.on_disconnect = on_disconnect
        c.on_message = on_message

        self._mqtt_client = c

        # Connect in background
        def loop():
            try:
                c.connect(self.mqtt_host, self.mqtt_port, keepalive=30)
                c.loop_forever(retry_first_connection=True)
            except Exception as e:
                with self._lock:
                    self._sensor.ok = False
                    self._sensor.err = f"mqtt_loop: {e}"

        threading.Thread(target=loop, daemon=True).start()

    # ---------- Routes ----------
    def _setup_routes(self):
        @self.app.get("/health")
        def health():
            return jsonify({"ok": True, "ts": time.time()})

        @self.app.get("/status")
        def status():
            with self._lock:
                return jsonify({
                    "ok": True,
                    "ts": time.time(),
                    "mqtt": {
                        "ok": self._sensor.ok,
                        "dt_ms": self._sensor.dt_ms,
                        "last_ts": self._sensor.last_ts,
                        "err": self._sensor.err,
                        "topic": self.mqtt_topic,
                    },
                    "dist_cm": self._sensor.dist_cm,
                    "temp_c": self._sensor.temp_c,
                    "humid": self._sensor.humid,
                    "toggles": {
                        "listening": self._listen_on,
                        "auto_move": self._auto_on,
                    },
                    "raw": self._sensor.raw,  # optional debug
                })

        @self.app.post("/toggle_listen")
        def toggle_listen():
            with self._lock:
                self._listen_on = not self._listen_on
                # rule: bật listening -> tắt auto move luôn để robot đứng yên
                if self._listen_on:
                    self._auto_on = False
                return jsonify({"ok": True, "listening": self._listen_on, "auto_move": self._auto_on})

        @self.app.post("/toggle_auto")
        def toggle_auto():
            with self._lock:
                self._auto_on = not self._auto_on
                # rule: bật auto -> tắt listening (đỡ tốn pin + tránh xung đột)
                if self._auto_on:
                    self._listen_on = False
                return jsonify({"ok": True, "auto_move": self._auto_on, "listening": self._listen_on})

        @self.app.get("/cmd")
        def cmd():
            move = (request.args.get("move") or "STOP").upper()
            if self.on_manual_cmd:
                try:
                    self.on_manual_cmd(move)
                except Exception:
                    pass
            return jsonify({"ok": True, "move": move})

        @self.app.get("/mjpeg")
        def mjpeg():
            def gen():
                black = None
                try:
                    import cv2
                    import numpy as np
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
    .btn-on { background:#1f7a3a; color:#fff; }
    .btn-off { background:#444; color:#fff; }
    .btn-warn { background:#8a1f1f; color:#fff; }
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
    </div>

    <div style="width:580px;">
      <div class="card">
        <b>Quick Status (MQTT Direct)</b>
        <div class="kv">
          <div class="pill">Dist(cm): <span id="dist">...</span></div>
          <div class="pill">Temp(°C): <span id="t">...</span></div>
          <div class="pill">Hum(%): <span id="h">...</span></div>
          <div class="pill">MQTT: <span id="mq">...</span></div>
        </div>
        <div class="err" id="err"></div>
      </div>

      <div class="card">
        <div><b>Modes</b></div>
        <div class="kv">
          <button id="btnListen" class="btn-off" onclick="toggleListen()">Active Listening: OFF</button>
          <button id="btnAuto" class="btn-off" onclick="toggleAuto()">Auto Move: OFF</button>
        </div>
        <div style="margin-top:8px; color:#aaa; font-size:13px;">
          *Listening ON → robot phải đứng yên và Auto Move sẽ tự tắt.
        </div>
      </div>

      <div class="card">
        <div><b>Manual Control</b></div>
        <div>
          <button onclick="cmd('FORWARD')">FORWARD</button>
        </div>
        <div>
          <button onclick="cmd('TURN_LEFT')">LEFT</button>
          <button class="btn-warn" onclick="cmd('STOP')">STOP</button>
          <button onclick="cmd('TURN_RIGHT')">RIGHT</button>
        </div>
        <div>
          <button onclick="cmd('BACK')">BACK</button>
        </div>
      </div>

      <div class="card">
        <div><b>Status JSON</b></div>
        <pre id="status">loading...</pre>
      </div>
    </div>
  </div>

<script>
function setBtn(el, on, labelOn, labelOff){
  if(on){
    el.classList.remove('btn-off');
    el.classList.add('btn-on');
    el.textContent = labelOn;
  }else{
    el.classList.remove('btn-on');
    el.classList.add('btn-off');
    el.textContent = labelOff;
  }
}

async function refresh(){
  const errEl = document.getElementById('err');
  try{
    const r = await fetch('/status', {cache:'no-store'});
    const j = await r.json();

    document.getElementById('status').textContent = JSON.stringify(j, null, 2);

    const mqttOk = j?.mqtt?.ok === true;
    const mq = mqttOk ? ('OK dt=' + (j.mqtt.dt_ms ?? 'NA') + 'ms') : ('ERR');
    document.getElementById('mq').textContent = mq;

    document.getElementById('dist').textContent = (j.dist_cm ?? 'NA');
    document.getElementById('t').textContent = (j.temp_c ?? 'NA');
    document.getElementById('h').textContent = (j.humid ?? 'NA');

    const listenOn = j?.toggles?.listening === true;
    const autoOn = j?.toggles?.auto_move === true;

    setBtn(document.getElementById('btnListen'), listenOn,
           'Active Listening: ON', 'Active Listening: OFF');

    setBtn(document.getElementById('btnAuto'), autoOn,
           'Auto Move: ON', 'Auto Move: OFF');

    errEl.textContent = (j?.mqtt?.err ? ('MQTT ERROR: ' + j.mqtt.err) : '');
  }catch(e){
    errEl.textContent = 'FETCH ERROR: ' + e;
  }
  setTimeout(refresh, 250);
}

async function cmd(m){
  await fetch('/cmd?move=' + m, {cache:'no-store'});
}

async function toggleListen(){
  await fetch('/toggle_listen', {method:'POST'});
}

async function toggleAuto(){
  await fetch('/toggle_auto', {method:'POST'});
}

refresh();
</script>
</body>
</html>
"""

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True)
