#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from typing import Optional, Callable, Any, Dict

import cv2
from flask import Flask, Response, jsonify, request


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Matthew Robot Dashboard</title>
  <style>
    body{margin:0;font-family:Arial;background:#0b0c10;color:#e5e7eb}
    .wrap{padding:18px}
    .title{font-size:34px;font-weight:800;margin:0 0 12px 0}
    .grid{display:grid;grid-template-columns: 1.6fr 1fr;gap:18px}
    .card{background:#111827;border:1px solid #1f2937;border-radius:16px;padding:14px;box-shadow:0 6px 20px rgba(0,0,0,.35)}
    .card h2{margin:0 0 10px 0;font-size:18px}
    .video{width:100%;border-radius:14px;border:1px solid #374151;background:#000}
    .chiprow{display:flex;gap:10px;flex-wrap:wrap}
    .chip{padding:10px 12px;border-radius:999px;border:1px solid #374151;background:#0f172a;font-weight:700}
    .chip.ok{border-color:#065f46;background:#052e2b}
    .chip.bad{border-color:#7f1d1d;background:#2a0b0b}
    pre{white-space:pre-wrap;word-break:break-word;background:#0b1220;border:1px solid #374151;border-radius:12px;padding:12px;margin:0}
    .btn{padding:10px 12px;border-radius:12px;border:1px solid #334155;background:#0f172a;color:#e5e7eb;cursor:pointer;font-weight:700}
    .btn.on{background:#14532d;border-color:#166534}
    @media(max-width:980px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="title">Matthew Robot Dashboard</div>
  <div class="grid">
    <div class="card">
      <h2>Live Camera</h2>
      <img class="video" id="cam" src="/mjpeg" />
    </div>

    <div>
      <div class="card" style="margin-bottom:18px">
        <h2>Quick Status</h2>
        <div class="chiprow">
          <div class="chip" id="chipObstacle">OBSTACLE: ...</div>
          <div class="chip" id="chipZone">ZONE: ...</div>
        </div>
      </div>

      <div class="card" style="margin-bottom:18px">
        <h2>Modes</h2>
        <button class="btn" id="btnHand">Hand Command: ...</button>
      </div>

      <div class="card">
        <h2>Status JSON</h2>
        <pre id="status">{}</pre>
      </div>
    </div>
  </div>
</div>

<script>
async function refresh(){
  try{
    const r = await fetch("/api/status");
    const js = await r.json();

    const obs = (js.obstacle && js.obstacle.label) ? js.obstacle.label : "no obstacle";
    const zone = (js.obstacle && js.obstacle.zone) ? js.obstacle.zone : "NONE";

    const chipObs = document.getElementById("chipObstacle");
    const chipZone = document.getElementById("chipZone");

    const yes = (obs === "yes have obstacle");
    chipObs.textContent = "OBSTACLE: " + (yes ? "YES" : "NO");
    chipObs.className = "chip " + (yes ? "bad" : "ok");

    chipZone.textContent = "ZONE: " + zone;
    chipZone.className = "chip";

    const btn = document.getElementById("btnHand");
    const en = (js.hand && js.hand.enabled === true);
    btn.textContent = "Hand Command: " + (en ? "ON" : "OFF");
    btn.className = "btn " + (en ? "on" : "");

    document.getElementById("status").textContent = JSON.stringify(js, null, 2);
  }catch(e){}
}
setInterval(refresh, 450);
refresh();

document.getElementById("btnHand").onclick = async () => {
  try{
    const r = await fetch("/api/hand/toggle", {method:"POST"});
    await r.json();
    refresh();
  }catch(e){}
};
</script>
</body>
</html>
"""


class WebDashboard:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        get_frame_bgr: Optional[Callable[[], Any]] = None,
        get_obstacle_state: Optional[Callable[[], Dict[str, Any]]] = None,
        on_manual_cmd: Optional[Callable[[str], None]] = None,
        rotate180: bool = False,
        hand_command: Any = None,
    ):
        self.host = host
        self.port = int(port)
        self.get_frame_bgr = get_frame_bgr
        self.get_obstacle_state = get_obstacle_state
        self.on_manual_cmd = on_manual_cmd
        self.rotate180 = bool(rotate180)
        self.hand_command = hand_command

        self.app = Flask("web_dashboard")

        @self.app.get("/")
        def home():
            return Response(HTML, mimetype="text/html")

        @self.app.get("/mjpeg")
        def mjpeg():
            return Response(self._mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.get("/api/status")
        def api_status():
            return jsonify(self._status_json())

        @self.app.post("/api/hand/toggle")
        def hand_toggle():
            ok = False
            enabled = None
            try:
                if self.hand_command is not None and hasattr(self.hand_command, "set_enabled"):
                    # infer current enabled
                    cur = self._get_hand_enabled()
                    newv = (not cur) if cur is not None else True
                    self.hand_command.set_enabled(bool(newv))
                    enabled = bool(newv)
                    ok = True
            except Exception:
                ok = False
            return jsonify({"ok": ok, "enabled": enabled})

        @self.app.post("/api/manual")
        def manual_cmd():
            cmd = (request.json or {}).get("cmd", "")
            if self.on_manual_cmd:
                try:
                    self.on_manual_cmd(str(cmd))
                except Exception:
                    pass
            return jsonify({"ok": True})

    def _get_hand_enabled(self) -> Optional[bool]:
        hc = self.hand_command
        if hc is None:
            return None
        for attr in ("enabled", "is_enabled"):
            if hasattr(hc, attr):
                v = getattr(hc, attr)
                if isinstance(v, bool):
                    return v
        if hasattr(hc, "get_status"):
            try:
                st = hc.get_status()
                if isinstance(st, dict) and "enabled" in st:
                    return bool(st["enabled"])
            except Exception:
                pass
        return None

    def _get_hand_status(self) -> Dict[str, Any]:
        hc = self.hand_command
        if hc is None:
            return {"enabled": False, "action": None, "fps": None, "finger_count": None}
        if hasattr(hc, "get_status"):
            try:
                st = hc.get_status()
                if isinstance(st, dict):
                    return st
            except Exception:
                pass
        # fallback minimal
        return {"enabled": bool(self._get_hand_enabled() or False), "action": None, "fps": None, "finger_count": None}

    def _status_json(self) -> Dict[str, Any]:
        obs = {"label": "no obstacle", "zone": "NONE"}
        if self.get_obstacle_state:
            try:
                o = self.get_obstacle_state()
                if isinstance(o, dict):
                    obs = {
                        "label": o.get("label", "no obstacle"),
                        "zone": o.get("zone", "NONE"),
                        "ts": o.get("ts", 0.0),
                        "stripe": o.get("stripe", None),
                    }
            except Exception:
                pass

        return {
            "ts": time.time(),
            "obstacle": obs,
            "hand": self._get_hand_status(),
        }

    def _mjpeg_generator(self):
        while True:
            frame = None
            try:
                if self.get_frame_bgr:
                    frame = self.get_frame_bgr()
            except Exception:
                frame = None

            if frame is None:
                time.sleep(0.08)
                continue

            if self.rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                time.sleep(0.03)
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")

            time.sleep(0.03)

    def run(self):
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
