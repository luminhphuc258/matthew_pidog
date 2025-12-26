#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from typing import Optional, Callable, Any, Dict

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
    .chip{padding:10px 12px;border-radius:999px;border:1px solid #374151;background:#0f172a;font-weight:800}
    .chip.ok{border-color:#065f46;background:#052e2b}
    .chip.bad{border-color:#7f1d1d;background:#2a0b0b}
    pre{white-space:pre-wrap;word-break:break-word;background:#0b1220;border:1px solid #374151;border-radius:12px;padding:12px;margin:0}
    .btn{padding:10px 12px;border-radius:12px;border:1px solid #334155;background:#0f172a;color:#e5e7eb;cursor:pointer;font-weight:900}
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
        get_jpeg: Optional[Callable[[], Optional[bytes]]] = None,
        get_obstacle_state: Optional[Callable[[], Dict[str, Any]]] = None,
        avoid_obstacle: Any = None,
        on_manual_cmd: Optional[Callable[[str], None]] = None,
        rotate180: bool = False,
        hand_command: Any = None,
    ):
        self.host = host
        self.port = int(port)

        self.get_frame_bgr = get_frame_bgr
        self.get_jpeg = get_jpeg

        self.get_obstacle_state = get_obstacle_state
        self.avoid_obstacle = avoid_obstacle

        self.on_manual_cmd = on_manual_cmd
        self.rotate180 = bool(rotate180)
        self.hand_command = hand_command

        self.app = Flask("web_dashboard")

        @self.app.get("/")
        def home():
            return Response(HTML, mimetype="text/html")

        @self.app.get("/mjpeg")
        def mjpeg():
            return Response(self._mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.get("/api/status")
        def api_status():
            return jsonify(self._status_json())

        @self.app.post("/api/hand/toggle")
        def hand_toggle():
            enabled = None
            ok = False
            try:
                cur = self._hand_enabled()
                newv = (not cur) if cur is not None else True
                enabled = self._hand_set_enabled(bool(newv))
                ok = (enabled is not None)
            except Exception:
                ok = False
            return jsonify({"ok": ok, "enabled": enabled})

        # ✅ ActiveListening gọi endpoint này (port 8000 localhost)
        @self.app.get("/take_gesture_meaning")
        def take_gesture_meaning():
            st = self._hand_status()
            # trả action nếu có
            return jsonify({
                "ok": True,
                "enabled": bool(st.get("enabled", False)),
                "action": st.get("action", None),
                "finger_count": st.get("finger_count", None),
                "fps": st.get("fps", None),
                "ts": time.time(),
            })

        @self.app.post("/api/manual")
        def manual_cmd():
            cmd = (request.json or {}).get("cmd", "")
            if self.on_manual_cmd:
                try:
                    self.on_manual_cmd(str(cmd))
                except Exception:
                    pass
            return jsonify({"ok": True})

    # -------------------------
    # Hand state helpers
    # -------------------------
    def _to_bool(self, v) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "on", "enabled"):
                return True
            if s in ("0", "false", "no", "off", "disabled"):
                return False
        return None

    def _hand_enabled(self) -> Optional[bool]:
        hc = self.hand_command
        if hc is None:
            return None

        for attr in ("enabled", "is_enabled", "_enabled"):
            if hasattr(hc, attr):
                try:
                    b = self._to_bool(getattr(hc, attr))
                    if b is not None:
                        return b
                except Exception:
                    pass

        if hasattr(hc, "get_status"):
            try:
                st = hc.get_status()
                if isinstance(st, dict):
                    b = self._to_bool(st.get("enabled", None))
                    if b is not None:
                        return b
            except Exception:
                pass

        return None

    def _hand_set_enabled(self, v: bool) -> Optional[bool]:
        hc = self.hand_command
        if hc is None:
            return None
        if hasattr(hc, "set_enabled"):
            try:
                hc.set_enabled(bool(v))
                return bool(v)
            except Exception:
                return None
        return None

    def _hand_status(self) -> Dict[str, Any]:
        hc = self.hand_command
        if hc is None:
            return {"enabled": False, "action": None, "finger_count": None, "fps": None}

        if hasattr(hc, "get_status"):
            try:
                st = hc.get_status()
                if isinstance(st, dict):
                    en = self._to_bool(st.get("enabled", None))
                    if en is None:
                        en = bool(self._hand_enabled() or False)
                    return {
                        "enabled": bool(en),
                        "action": st.get("action", None),
                        "finger_count": st.get("finger_count", None),
                        "fps": st.get("fps", None),
                    }
            except Exception:
                pass

        return {"enabled": bool(self._hand_enabled() or False), "action": None, "finger_count": None, "fps": None}

    # -------------------------
    # Obstacle status
    # -------------------------
    def _obstacle_state(self) -> Dict[str, Any]:
        # ưu tiên get_obstacle_state()
        if self.get_obstacle_state:
            try:
                o = self.get_obstacle_state()
                if isinstance(o, dict):
                    return {
                        "label": o.get("label", "no obstacle"),
                        "zone": o.get("zone", "NONE"),
                        "ts": o.get("ts", 0.0),
                        "shape": o.get("shape", None),
                        "orientation": o.get("orientation", None),
                    }
            except Exception:
                pass

        # fallback nếu avoid_obstacle có get_state
        ao = self.avoid_obstacle
        if ao is not None and hasattr(ao, "get_state"):
            try:
                o = ao.get_state()
                if isinstance(o, dict):
                    return {
                        "label": o.get("label", "no obstacle"),
                        "zone": o.get("zone", "NONE"),
                        "ts": o.get("ts", 0.0),
                    }
            except Exception:
                pass

        return {"label": "no obstacle", "zone": "NONE", "ts": 0.0}

    def _status_json(self) -> Dict[str, Any]:
        return {
            "ts": time.time(),
            "obstacle": self._obstacle_state(),
            "hand": self._hand_status(),
        }

    # -------------------------
    # MJPEG generator (giống code cũ của bạn)
    # -------------------------
    def _mjpeg_gen(self):
        import numpy as np
        import cv2

        black = None
        try:
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
                        if self.rotate180:
                            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)

                        # obstacle overlay
                        if self.avoid_obstacle is not None:
                            try:
                                frame_bgr = self.avoid_obstacle.draw_overlay(frame_bgr)
                            except Exception:
                                pass

                        # ✅ draw hand overlay (đúng như code cũ)
                        if (self._hand_enabled() is True) and (self.hand_command is not None):
                            try:
                                # QUAN TRỌNG: gọi đúng hàm bạn dùng trước đây
                                if hasattr(self.hand_command, "draw_on_frame"):
                                    frame_bgr = self.hand_command.draw_on_frame(frame_bgr)
                                elif hasattr(self.hand_command, "draw_on_frame_bgr"):
                                    frame_bgr = self.hand_command.draw_on_frame_bgr(frame_bgr)
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

    def run(self):
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
