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
    strength: Optional[float] = None

    src_ts_ms: Optional[int] = None
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


def _to_int(x):
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


class WebDashboard:
    """
    - /            HTML dashboard
    - /mjpeg       live video
    - /status      mqtt sensor + toggles + (avoid_obstacle state) + hand
    - /cmd         manual command
    - /toggle_listen      listen-only
    - /toggle_auto        auto-move only
    - /toggle_listen_run  listen + auto-move (test)
    - /toggle_hand        NEW: hand command
    - /health
    """

    def __init__(
        self,
        host="0.0.0.0",
        port=8000,

        get_jpeg: Optional[Callable[[], Optional[bytes]]] = None,
        get_frame_bgr: Optional[Callable[[], Any]] = None,

        avoid_obstacle: Optional[Any] = None,
        on_manual_cmd: Optional[Callable[[str], None]] = None,

        rotate180: bool = True,

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

        # ✅ NEW
        hand_command: Optional[Any] = None,
    ):
        self.host, self.port = host, port

        self.get_jpeg = get_jpeg
        self.get_frame_bgr = get_frame_bgr
        self.avoid_obstacle = avoid_obstacle
        self.on_manual_cmd = on_manual_cmd
        self.hand_command = hand_command

        self.rotate180 = bool(rotate180)

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
        self._listen_run_on = False

        # ✅ NEW
        self._hand_on = False

        self.app = Flask(__name__)
        self._setup_routes()

        self._mqtt_client = None
        if self.mqtt_enable:
            self._start_mqtt()

    def is_hand_on(self) -> bool:
        with self._lock:
            return bool(self._hand_on)

    # ===== MQTT =====
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
            c.tls_set()
            if self.mqtt_insecure:
                c.tls_insecure_set(True)

        def on_connect(client, userdata, flags, rc):
            if self.mqtt_debug:
                print("[WebDash MQTT] connect rc=", rc, flush=True)
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
                print("[WebDash MQTT] disconnect rc=", rc, flush=True)
            with self._lock:
                self._sensor.ok = False
                self._sensor.err = f"disconnect rc={rc}"

        def on_message(client, userdata, msg):
            now = time.time()
            data = None
            err = None
            try:
                s = msg.payload.decode("utf-8", errors="ignore").strip()
                data = json.loads(s)
                if not isinstance(data, dict):
                    data = {"value": data}
            except Exception as e:
                err = f"parse_json: {e}"
                data = None

            with self._lock:
                if self._sensor.last_ts > 0:
                    self._sensor.dt_ms = int((now - self._sensor.last_ts) * 1000)
                self._sensor.last_ts = now

                if data is not None:
                    dist = _pick(data, ["uart_dist_cm", "lidar_cm", "dist_cm", "distance_cm", "distance", "dist", "range_cm"])
                    strength = _pick(data, ["uart_strength", "strength"])
                    temp = _pick(data, ["temp_c", "uart_temp_c", "temperature_c", "temperature", "temp", "t"])
                    hum = _pick(data, ["humid", "uart_humid", "humidity", "hum", "h"])
                    ts_ms = _pick(data, ["ts_ms", "timestamp_ms", "ms"])

                    self._sensor.dist_cm = _to_float(dist)
                    self._sensor.strength = _to_float(strength)
                    self._sensor.temp_c = _to_float(temp)
                    self._sensor.humid = _to_float(hum)
                    self._sensor.src_ts_ms = _to_int(ts_ms)

                    self._sensor.err = None
                    self._sensor.ok = True
                else:
                    self._sensor.err = err
                    self._sensor.ok = False

        c.on_connect = on_connect
        c.on_disconnect = on_disconnect
        c.on_message = on_message

        self._mqtt_client = c

        def loop():
            try:
                c.connect(self.mqtt_host, self.mqtt_port, keepalive=30)
                c.loop_forever(retry_first_connection=True)
            except Exception as e:
                with self._lock:
                    self._sensor.ok = False
                    self._sensor.err = f"mqtt_loop: {e}"

        threading.Thread(target=loop, daemon=True).start()

    # ===== Routes =====
    def _setup_routes(self):
        @self.app.get("/health")
        def health():
            return jsonify({"ok": True, "ts": time.time()})

        @self.app.get("/status")
        def status():
            avoid_state = None
            if self.avoid_obstacle is not None:
                try:
                    avoid_state = self.avoid_obstacle.get_state()
                except Exception:
                    avoid_state = {"error": "avoid_obstacle.get_state failed"}

            hand_state = None
            if self.hand_command is not None:
                try:
                    hand_state = self.hand_command.get_last()
                except Exception:
                    hand_state = {"error": "hand_command.get_last failed"}

            with self._lock:
                return jsonify({
                    "ok": True,
                    "ts": time.time(),
                    "mqtt": {
                        "ok": self._sensor.ok,
                        "dt_ms": self._sensor.dt_ms,
                        "last_ts": self._sensor.last_ts,
                        "src_ts_ms": self._sensor.src_ts_ms,
                        "err": self._sensor.err,
                        "topic": self.mqtt_topic,
                    },

                    "lidar_cm": self._sensor.dist_cm,
                    "lidar_strength": self._sensor.strength,
                    "temp_c": self._sensor.temp_c,
                    "humid": self._sensor.humid,

                    "toggles": {
                        "listening": self._listen_on,
                        "auto_move": self._auto_on,
                        "listen_run": self._listen_run_on,
                        "hand_cmd": self._hand_on,  # ✅ NEW
                    },
                    "avoid_obstacle": avoid_state,
                    "hand": hand_state,           # ✅ NEW
                    "video": {"rotate180": self.rotate180},
                })

        @self.app.post("/toggle_hand")
        def toggle_hand():
            with self._lock:
                self._hand_on = not self._hand_on
                on = self._hand_on

            if self.hand_command is not None:
                try:
                    self.hand_command.set_enabled(on)
                except Exception:
                    pass

            return jsonify({"ok": True, "toggles": {"hand_cmd": on}})

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
                    import numpy as np
                    import cv2
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if ok:
                        black = buf.tobytes()
                except Exception:
                    black = None

                use_bgr = self.get_frame_bgr is not None

                while True:
                    try:
                        if use_bgr:
                            frame_bgr = self.get_frame_bgr()
                            if frame_bgr is None:
                                frame_bytes = black
                            else:
                                import cv2
                                if self.rotate180:
                                    frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)

                                if self.avoid_obstacle is not None:
                                    try:
                                        frame_bgr = self.avoid_obstacle.draw_overlay(frame_bgr)
                                    except Exception:
                                        pass

                                # ✅ NEW: draw hand overlay only when toggle on
                                hand_on = False
                                with self._lock:
                                    hand_on = self._hand_on

                                if hand_on and self.hand_command is not None:
                                    try:
                                        frame_bgr = self.hand_command.draw_on_frame(frame_bgr)
                                    except Exception:
                                        pass

                                ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                                frame_bytes = buf.tobytes() if ok else black

                            if frame_bytes:
                                yield (b"--frame\r\n"
                                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                            else:
                                time.sleep(0.05)
                        else:
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
    .btn-on { background:#1f7a3a; color:#fff; }
    .btn-off { background:#444; color:#fff; }
    .btn-warn { background:#8a1f1f; color:#fff; }
    .btn-test-on { background:#2d4f9e; color:#fff; }
    .btn-test-off { background:#1f2f55; color:#fff; }
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
        <b>Quick Status</b>
        <div class="kv">
          <div class="pill">LiDAR(cm): <span id="lidar">...</span></div>
          <div class="pill">Strength: <span id="str">...</span></div>
          <div class="pill">Temp(°C): <span id="t">...</span></div>
          <div class="pill">Hum(%): <span id="h">...</span></div>
          <div class="pill">HAND: <span id="hand">...</span></div>
        </div>
        <div class="err" id="err"></div>
      </div>

      <div class="card">
        <div><b>Modes</b></div>
        <div class="kv">
          <button id="btnHand" class="btn-off" onclick="toggleHand()">Hand Command: OFF</button>
        </div>
      </div>

      <div class="card">
        <div><b>Status JSON</b></div>
        <pre id="status">loading...</pre>
      </div>
    </div>
  </div>

<script>
function setBtn(el, on, labelOn, labelOff, clsOn, clsOff){
  el.className = on ? clsOn : clsOff;
  el.textContent = on ? labelOn : labelOff;
}

async function refresh(){
  const errEl = document.getElementById('err');
  try{
    const r = await fetch('/status', {cache:'no-store'});
    const j = await r.json();

    document.getElementById('status').textContent = JSON.stringify(j, null, 2);

    document.getElementById('lidar').textContent = (j.lidar_cm ?? 'NA');
    document.getElementById('str').textContent   = (j.lidar_strength ?? 'NA');
    document.getElementById('t').textContent     = (j.temp_c ?? 'NA');
    document.getElementById('h').textContent     = (j.humid ?? 'NA');

    const handOn = j?.toggles?.hand_cmd === true;
    const gest = j?.hand?.gesture ?? 'NA';
    const fps = j?.hand?.fps ?? null;
    document.getElementById('hand').textContent = gest + (fps ? (' ('+fps.toFixed(1)+'fps)') : '');

    setBtn(document.getElementById('btnHand'), handOn,
      'Hand Command: ON', 'Hand Command: OFF', 'btn-on', 'btn-off');

    errEl.textContent = (j?.mqtt?.err ? ('MQTT ERROR: ' + j.mqtt.err) : '');
  }catch(e){
    errEl.textContent = 'FETCH ERROR: ' + e;
  }
  setTimeout(refresh, 250);
}

async function toggleHand(){
  await fetch('/toggle_hand', {method:'POST'});
}

refresh();
</script>
</body>
</html>
"""

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True)
