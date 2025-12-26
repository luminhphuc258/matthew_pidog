#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    .btn{padding:10px 12px;border-radius:12px;border:1px solid #334155;background:#0f172a;color:#e5e7eb;cursor:pointer;font-weight:800}
    .btn.on{background:#14532d;border-color:#166534}
    .btn.off{background:#0f172a;border-color:#334155}
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
    btn.className = "btn " + (en ? "on" : "off");

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
        draw_hand_overlay: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.get_frame_bgr = get_frame_bgr
        self.get_obstacle_state = get_obstacle_state
        self.on_manual_cmd = on_manual_cmd
        self.rotate180 = bool(rotate180)
        self.hand_command = hand_command
        self.draw_hand_overlay = bool(draw_hand_overlay)

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
                enabled = self._toggle_hand_enabled()
                ok = (enabled is not None)
            except Exception:
                ok = False
            return jsonify({"ok": ok, "enabled": enabled})

        @self.app.post("/api/hand/set")
        def hand_set():
            ok = False
            enabled = None
            try:
                js = request.json or {}
                val = js.get("enabled", None)
                if val is None:
                    return jsonify({"ok": False, "error": "missing enabled"}), 400
                enabled = self._set_hand_enabled(bool(val))
                ok = (enabled is not None)
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

    # -----------------------------
    # Hand helpers (robust)
    # -----------------------------
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

    def _get_hand_enabled(self) -> Optional[bool]:
        hc = self.hand_command
        if hc is None:
            return None

        # direct attrs
        for attr in ("enabled", "is_enabled", "_enabled"):
            if hasattr(hc, attr):
                try:
                    b = self._to_bool(getattr(hc, attr))
                    if b is not None:
                        return b
                except Exception:
                    pass

        # get_status dict
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

    def _set_hand_enabled(self, enabled: bool) -> Optional[bool]:
        hc = self.hand_command
        if hc is None:
            return None

        # prefer set_enabled
        if hasattr(hc, "set_enabled"):
            try:
                hc.set_enabled(bool(enabled))
                return bool(enabled)
            except Exception:
                pass

        # fallback: try enable()/disable()
        if bool(enabled) and hasattr(hc, "enable"):
            try:
                hc.enable()
                return True
            except Exception:
                pass
        if (not bool(enabled)) and hasattr(hc, "disable"):
            try:
                hc.disable()
                return False
            except Exception:
                pass

        return self._get_hand_enabled()

    def _toggle_hand_enabled(self) -> Optional[bool]:
        cur = self._get_hand_enabled()
        if cur is None:
            # unknown -> force ON
            return self._set_hand_enabled(True)
        return self._set_hand_enabled(not cur)

    def _normalize_hand_status(self, st: Dict[str, Any]) -> Dict[str, Any]:
        enabled = self._to_bool(st.get("enabled", None))
        if enabled is None:
            enabled = bool(self._get_hand_enabled() or False)

        def _num(v):
            return v if isinstance(v, (int, float)) else None

        return {
            "enabled": bool(enabled),
            "action": st.get("action", None),
            "finger_count": st.get("finger_count", None),
            "fps": _num(st.get("fps", None)),
        }

    def _get_hand_status(self) -> Dict[str, Any]:
        hc = self.hand_command
        if hc is None:
            return {"enabled": False, "action": None, "fps": None, "finger_count": None}

        if hasattr(hc, "get_status"):
            try:
                st = hc.get_status()
                if isinstance(st, dict):
                    return self._normalize_hand_status(st)
            except Exception:
                pass

        # fallback minimal
        return {"enabled": bool(self._get_hand_enabled() or False), "action": None, "fps": None, "finger_count": None}

    # -----------------------------
    # Draw hand overlay (restore)
    # -----------------------------
    def _apply_hand_overlay(self, frame_bgr):
        hc = self.hand_command
        if hc is None:
            return frame_bgr

        # Try common draw functions from HandCommand
        for meth in ("get_debug_draw", "draw_debug", "annotate", "draw_overlay", "render_debug"):
            if hasattr(hc, meth):
                fn = getattr(hc, meth)
                try:
                    out = fn(frame_bgr)
                    if isinstance(out, type(frame_bgr)) and getattr(out, "shape", None) is not None:
                        return out
                except Exception:
                    pass

        # Fallback: draw HUD text only
        try:
            st = self._get_hand_status()
            en = st.get("enabled", False)
            act = st.get("action") or "NA"
            fc = st.get("finger_count")
            fps = st.get("fps")

            if fps is not None:
                txt = f"HAND: {'ON' if en else 'OFF'}  {act}  fingers:{fc if fc is not None else 'NA'}  ({fps:.1f}fps)"
            else:
                txt = f"HAND: {'ON' if en else 'OFF'}  {act}  fingers:{fc if fc is not None else 'NA'}"

            cv2.putText(frame_bgr, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        except Exception:
            pass

        return frame_bgr

    # -----------------------------
    # Status JSON
    # -----------------------------
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
                        "shape": o.get("shape", None),
                        "orientation": o.get("orientation", None),
                    }
            except Exception:
                pass

        return {
            "ts": time.time(),
            "obstacle": obs,
            "hand": self._get_hand_status(),
        }

    # -----------------------------
    # MJPEG
    # -----------------------------
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

            # IMPORTANT: overlay BEFORE rotate (so coords still match raw)
            if self.draw_hand_overlay:
                try:
                    frame = self._apply_hand_overlay(frame)
                except Exception:
                    pass

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
