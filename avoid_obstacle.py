#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import requests


@dataclass
class AvoidCfg:
    cam_dev: str = "/dev/video0"
    cam_w: int = 640
    cam_h: int = 480
    cam_fps: int = 30

    loop_hz: float = 15.0

    # near-field ROI
    roi_y_start_ratio: float = 0.55
    roi_y_end_ratio: float = 1.00

    sector_n: int = 9

    # distance thresholds (cm)
    trigger_cm: float = 140.0       # gọi GPT khi bắt đầu gần
    hard_stop_cm: float = 35.0      # cực gần -> stop
    min_valid_cm: float = 2.0
    max_valid_cm: float = 800.0

    min_trigger_interval_sec: float = 1.5
    plan_ttl_sec: float = 6.0

    server_url: str = "https://embeddedprogramming-healtheworldserver.up.railway.app/avoid_obstacle_vision"
    jpeg_quality: int = 60
    send_w: int = 320
    send_h: int = 180

    # ===== floor model (HSV) =====
    floor_sample_h: int = 18
    hsv_tolerance: Tuple[int, int, int] = (16, 70, 80)

    # EMA để tránh “học nhầm sàn”
    floor_ema_alpha: float = 0.12
    floor_update_min_conf: float = 0.55

    # corridor detector
    corridor_min_width_ratio: float = 0.18   # lối đi tối thiểu (theo bề ngang ROI)
    corridor_prefer_center_bias: float = 0.35  # ưu tiên gần giữa

    # gating gpt
    gpt_min_conf: float = 0.55               # conf thấp -> fallback local
    gpt_timeout_sec: float = 7.0


class AvoidObstacle:
    """
    Context-aware walkway navigation:
    - Local: corridor detection from floor mask (robust hơn sector mean)
    - Remote (NodeJS GPT): can return walkway polygon/center + obstacles
    - Fusion: choose best path for human walkway
    """

    def __init__(
        self,
        get_ultrasonic_cm: Callable[[], Optional[float]],
        cfg: AvoidCfg = AvoidCfg(),
        session: Optional[requests.Session] = None,
        get_frame_bgr: Optional[Callable[[], Any]] = None,
        rotate180: bool = True,
        get_lidar_strength: Optional[Callable[[], Optional[float]]] = None,
    ):
        self.get_ultrasonic_cm = get_ultrasonic_cm
        self.get_lidar_strength = get_lidar_strength
        self.cfg = cfg
        self.http = session or requests.Session()

        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self.get_frame_bgr = get_frame_bgr
        self.rotate180 = bool(rotate180)

        self._lock = threading.Lock()
        self._last_frame = None
        self._last_roi = None
        self._last_mask_floor = None  # for debug overlay
        self._last_local = {}
        self._last_plan: Dict[str, Any] = {}
        self._last_plan_ts = 0.0
        self._last_trigger_ts = 0.0

        # floor EMA state
        self._floor_hsv_ema = None  # np.array([H,S,V], float)

        # decision output for robot
        self._decision = {
            "mode": "local",
            "turn": 0.0,     # -1 left .. +1 right
            "speed": 0.0,    # 0..1
            "reason": "init",
        }

    # ------------------- Public API -------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if self.get_frame_bgr is None:
            self._open_camera()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.get_frame_bgr is None:
            self._close_camera()

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            plan_age = now - self._last_plan_ts if self._last_plan_ts else None
            return {
                "local": self._last_local,
                "plan": self._last_plan,
                "plan_age_sec": plan_age,
                "decision": dict(self._decision),
                "rotate180": self.rotate180,
            }

    def get_decision(self) -> Dict[str, Any]:
        """Main robot loop can read this and convert to motion command."""
        with self._lock:
            return dict(self._decision)

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        y0 = int(self.cfg.roi_y_start_ratio * h)
        y1 = int(self.cfg.roi_y_end_ratio * h)

        # ROI box
        cv2.rectangle(frame_bgr, (0, y0), (w - 1, y1 - 1), (200, 200, 200), 1)
        cv2.putText(frame_bgr, "ROI near-field", (8, max(18, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        with self._lock:
            local = dict(self._last_local or {})
            plan = dict(self._last_plan or {})
            plan_ts = self._last_plan_ts
            decision = dict(self._decision)
            mask_floor = self._last_mask_floor

        # show local corridor band
        cx = local.get("corridor_center_x", None)
        cw = local.get("corridor_width_px", None)
        conf = local.get("corridor_conf", None)

        if isinstance(cx, (int, float)) and isinstance(cw, (int, float)):
            cx = int(cx)
            half = int(cw / 2)
            x1 = max(0, cx - half)
            x2 = min(w - 1, cx + half)
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (x1, y0), (x2, y1), (0, 255, 0), -1)
            frame_bgr[:] = cv2.addWeighted(overlay, 0.12, frame_bgr, 0.88, 0)
            cv2.line(frame_bgr, (cx, y0), (cx, y1), (0, 255, 0), 2)

        # local HUD
        dist = local.get("lidar_cm", None)
        strg = local.get("lidar_strength", None)
        txt = f"LiDAR={dist}cm str={strg} corridor_conf={conf}"
        cv2.putText(frame_bgr, txt, (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # decision HUD
        dmode = decision.get("mode")
        turn = decision.get("turn")
        spd = decision.get("speed")
        reason = decision.get("reason")
        cv2.putText(frame_bgr, f"DECISION: {dmode} turn={turn:.2f} speed={spd:.2f}",
                    (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"reason: {reason}"[:70],
                    (8, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # GPT overlay (walkway_poly / safe_poly / obstacles)
        now = time.time()
        if plan and plan_ts and (now - plan_ts) <= self.cfg.plan_ttl_sec:
            roi_h = y1 - y0

            # walkway polygon in ROI coords: [[x,y],...]
            poly = plan.get("walkway_poly") or plan.get("safe_poly")
            if isinstance(poly, list) and len(poly) >= 3:
                pts = []
                for p in poly:
                    if isinstance(p, list) and len(p) == 2:
                        px, py = int(p[0]), int(p[1])
                        px = max(0, min(w - 1, px))
                        py = max(0, min(roi_h - 1, py))
                        pts.append([px, py + y0])
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    overlay = frame_bgr.copy()
                    cv2.fillPoly(overlay, [pts_np], (0, 200, 255))
                    frame_bgr[:] = cv2.addWeighted(overlay, 0.12, frame_bgr, 0.88, 0)
                    cv2.polylines(frame_bgr, [pts_np], True, (0, 200, 255), 2)

            obstacles = plan.get("obstacles", [])
            if isinstance(obstacles, list):
                for ob in obstacles[:12]:
                    bbox = ob.get("bbox", None)
                    label = str(ob.get("label", "obj"))
                    risk = ob.get("risk", None)
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x1, yb1, x2, yb2 = [int(v) for v in bbox]
                        x1 = max(0, min(w - 1, x1))
                        x2 = max(0, min(w - 1, x2))
                        yb1 = max(0, min(roi_h - 1, yb1))
                        yb2 = max(0, min(roi_h - 1, yb2))
                        cv2.rectangle(frame_bgr, (x1, y0 + yb1), (x2, y0 + yb2), (0, 150, 255), 2)
                        t = f"{label}"
                        if isinstance(risk, (int, float)):
                            t += f" r={risk:.2f}"
                        cv2.putText(frame_bgr, t, (x1, y0 + max(14, yb1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1, cv2.LINE_AA)

            conf_g = plan.get("confidence", None)
            best = plan.get("best_sector", None)
            cv2.putText(frame_bgr, f"GPT conf={conf_g} best={best}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        return frame_bgr

    # ------------------- Core Loop -------------------

    def _open_camera(self):
        self._cap = cv2.VideoCapture(self.cfg.cam_dev)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_h)
        self._cap.set(cv2.CAP_PROP_FPS, self.cfg.cam_fps)

    def _close_camera(self):
        try:
            if self._cap:
                self._cap.release()
        finally:
            self._cap = None

    def _loop(self):
        dt = 1.0 / max(1.0, self.cfg.loop_hz)
        while not self._stop.is_set():
            t0 = time.time()

            frame = None
            ok = False
            if self.get_frame_bgr is not None:
                try:
                    frame = self.get_frame_bgr()
                    ok = frame is not None
                except Exception:
                    frame = None
                    ok = False
            else:
                ok, frame = self._cap.read() if self._cap else (False, None)

            if ok and frame is not None:
                if self.rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                roi = self._extract_roi(frame)

                # distance + strength
                dist = self._read_distance_cm()
                strength = self._read_strength()

                # local context: floor mask -> corridor
                local = self._local_walkway_context(roi)

                local["lidar_cm"] = dist
                local["lidar_strength"] = strength
                local["ts"] = time.time()

                # compute decision (fusion)
                decision = self._decide(local, dist)

                with self._lock:
                    self._last_frame = frame
                    self._last_roi = roi
                    self._last_local = local
                    self._decision = decision

                # call GPT if needed
                if self._should_trigger(dist, local):
                    self._trigger_upload(roi, dist, local)

            elapsed = time.time() - t0
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _extract_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        y0 = int(self.cfg.roi_y_start_ratio * h)
        y1 = int(self.cfg.roi_y_end_ratio * h)
        return frame_bgr[y0:y1, 0:w].copy()

    def _read_distance_cm(self) -> Optional[float]:
        try:
            d = self.get_ultrasonic_cm()
            if d is None:
                return None
            d = float(d)
            if d < self.cfg.min_valid_cm or d > self.cfg.max_valid_cm:
                return None
            return d
        except Exception:
            return None

    def _read_strength(self) -> Optional[float]:
        if self.get_lidar_strength is None:
            return None
        try:
            s = self.get_lidar_strength()
            if s is None:
                return None
            return float(s)
        except Exception:
            return None

    # ========= Local walkway context =========

    def _local_walkway_context(self, roi_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Produce:
        - mask_floor
        - corridor_center_x, corridor_width_px, corridor_width_ratio
        - corridor_conf
        """
        h, w = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # 1) sample floor from 3 bands (left-mid-right) at bottom
        sh = min(self.cfg.floor_sample_h, h)
        band = hsv[h - sh:h, :, :]

        thirds = [
            band[:, 0:int(w*0.33), :],
            band[:, int(w*0.33):int(w*0.66), :],
            band[:, int(w*0.66):w, :],
        ]
        meds = []
        for b in thirds:
            if b.size > 0:
                meds.append(np.median(b.reshape(-1, 3), axis=0))
        med = np.median(np.stack(meds, axis=0), axis=0) if meds else np.median(band.reshape(-1, 3), axis=0)

        mh, ms, mv = float(med[0]), float(med[1]), float(med[2])

        # 2) EMA stabilize floor color (update only if last conf good)
        if self._floor_hsv_ema is None:
            self._floor_hsv_ema = np.array([mh, ms, mv], dtype=np.float32)
        else:
            # we don't know conf yet; do a provisional mask with current EMA
            pass

        # Build mask using EMA color
        mh2, ms2, mv2 = [float(x) for x in self._floor_hsv_ema]
        th, ts, tv = self.cfg.hsv_tolerance
        lo = np.array([max(0, mh2 - th), max(0, ms2 - ts), max(0, mv2 - tv)], dtype=np.uint8)
        hi = np.array([min(179, mh2 + th), min(255, ms2 + ts), min(255, mv2 + tv)], dtype=np.uint8)

        mask_floor = cv2.inRange(hsv, lo, hi)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_floor = cv2.morphologyEx(mask_floor, cv2.MORPH_OPEN, k, iterations=1)
        mask_floor = cv2.morphologyEx(mask_floor, cv2.MORPH_CLOSE, k, iterations=1)

        # Keep for debug overlay
        with self._lock:
            self._last_mask_floor = mask_floor.copy()

        # 3) Corridor detection: scan a strip near bottom of ROI
        # Take bottom 25% of ROI, find widest continuous floor segment.
        y_strip0 = int(h * 0.70)
        strip = mask_floor[y_strip0:h, :]  # 0/255
        # column score: how much floor in each column
        col_floor_ratio = np.mean(strip > 0, axis=0)  # 0..1
        # floor columns if ratio high
        is_floor_col = col_floor_ratio > 0.55

        # find longest run of True
        runs = []
        start = None
        for i, v in enumerate(is_floor_col):
            if v and start is None:
                start = i
            if (not v or i == w - 1) and start is not None:
                end = i if not v else i + 1
                runs.append((start, end))
                start = None

        # choose run with best score (width + center bias)
        best_run = None
        best_score = -1e9
        for (a, b) in runs:
            width = b - a
            width_ratio = width / max(1, w)
            if width_ratio < self.cfg.corridor_min_width_ratio:
                continue
            cx = (a + b) * 0.5
            center_bias = 1.0 - abs(cx - w*0.5) / (w*0.5)  # 0..1
            score = (width_ratio * 1.2) + (center_bias * self.cfg.corridor_prefer_center_bias)
            if score > best_score:
                best_score = score
                best_run = (a, b)

        if best_run is None:
            # fallback: use sector method but weaker
            scores = []
            sw = w // self.cfg.sector_n
            for i in range(self.cfg.sector_n):
                x0 = i * sw
                x1 = (i + 1) * sw if i < self.cfg.sector_n - 1 else w
                m = strip[:, x0:x1]
                scores.append(float(np.mean(m > 0)))
            best_sector = int(np.argmax(scores)) if scores else 0
            cx = int((best_sector + 0.5) * (w / self.cfg.sector_n))
            conf = float(max(scores)) if scores else 0.0
            width_px = int(w / self.cfg.sector_n)
            ctx = {
                "method": "sector_fallback",
                "scores": scores,
                "best_sector": best_sector,
                "corridor_center_x": cx,
                "corridor_width_px": width_px,
                "corridor_width_ratio": width_px / max(1, w),
                "corridor_conf": conf * 0.6,
                "floor_hsv_ema": [float(x) for x in self._floor_hsv_ema],
            }
            return ctx

        a, b = best_run
        cx = int((a + b) * 0.5)
        width_px = int(b - a)
        width_ratio = width_px / max(1, w)

        # corridor confidence: based on how solid floor is in that run + width
        solid = float(np.mean(col_floor_ratio[a:b])) if b > a else 0.0
        conf = float(0.55 * solid + 0.45 * min(1.0, width_ratio / 0.45))

        # Update EMA only if confidence high enough (avoid learning wrong thing)
        if conf >= self.cfg.floor_update_min_conf:
            alpha = float(self.cfg.floor_ema_alpha)
            self._floor_hsv_ema = (1.0 - alpha) * self._floor_hsv_ema + alpha * np.array([mh, ms, mv], dtype=np.float32)

        # compute sector index for compatibility
        best_sector = int(np.clip(cx / (w / self.cfg.sector_n), 0, self.cfg.sector_n - 1))

        return {
            "method": "corridor",
            "best_sector": best_sector,
            "corridor_center_x": cx,
            "corridor_width_px": width_px,
            "corridor_width_ratio": width_ratio,
            "corridor_conf": conf,
            "floor_hsv_ema": [float(x) for x in self._floor_hsv_ema],
        }

    # ========= Decision fusion =========

    def _decide(self, local: Dict[str, Any], dist_cm: Optional[float]) -> Dict[str, Any]:
        """
        Return: {turn, speed, mode, reason}
        turn: -1..+1 where -1 left +1 right
        """
        # Hard stop
        if dist_cm is not None and dist_cm <= self.cfg.hard_stop_cm:
            return {"mode": "hard_stop", "turn": 0.0, "speed": 0.0, "reason": f"hard_stop dist={dist_cm}"}

        # Try use GPT plan if fresh and confident
        with self._lock:
            plan = dict(self._last_plan or {})
            plan_ts = self._last_plan_ts

        now = time.time()
        if plan and plan_ts and (now - plan_ts) <= self.cfg.plan_ttl_sec:
            conf = plan.get("confidence", None)
            if isinstance(conf, (int, float)) and conf >= self.cfg.gpt_min_conf:
                # If server gives walkway_center_x in ROI coords
                wcx = plan.get("walkway_center_x", None)
                if isinstance(wcx, (int, float)):
                    # map to turn
                    roi_w = self.cfg.cam_w  # approx; we use full frame width for overlay, OK
                    turn = float(np.clip((wcx - roi_w * 0.5) / (roi_w * 0.5), -1.0, 1.0))
                    speed = 0.35 if dist_cm and dist_cm < 80 else 0.55
                    return {"mode": "gpt", "turn": turn, "speed": speed, "reason": "gpt walkway_center_x"}

                # else: server gives best_sector
                bs = plan.get("best_sector", None)
                if isinstance(bs, int):
                    # convert sector to turn
                    center_sector = (self.cfg.sector_n - 1) / 2.0
                    turn = float(np.clip((bs - center_sector) / center_sector, -1.0, 1.0))
                    speed = 0.35 if dist_cm and dist_cm < 80 else 0.55
                    return {"mode": "gpt", "turn": turn, "speed": speed, "reason": "gpt best_sector"}

        # fallback local corridor
        cx = local.get("corridor_center_x", None)
        conf = local.get("corridor_conf", None)

        if isinstance(cx, (int, float)):
            # turn based on corridor center relative to image center
            roi_w = self.cfg.cam_w
            turn = float(np.clip((float(cx) - roi_w * 0.5) / (roi_w * 0.5), -1.0, 1.0))
            # speed uses confidence & distance
            base = 0.45 + 0.35 * float(conf if isinstance(conf, (int, float)) else 0.0)
            if dist_cm is not None and dist_cm < 90:
                base *= 0.7
            return {"mode": "local", "turn": turn, "speed": float(np.clip(base, 0.15, 0.75)), "reason": "local corridor"}

        # last fallback: stop
        return {"mode": "unknown", "turn": 0.0, "speed": 0.0, "reason": "no corridor"}

    # ========= Trigger + Upload =========

    def _should_trigger(self, dist_cm: Optional[float], local: Dict[str, Any]) -> bool:
        now = time.time()
        if (now - self._last_trigger_ts) < self.cfg.min_trigger_interval_sec:
            return False

        # call GPT if near
        if dist_cm is not None and dist_cm <= self.cfg.trigger_cm:
            return True

        # call GPT if corridor is uncertain (scene complex)
        cconf = local.get("corridor_conf", 0.0)
        if isinstance(cconf, (int, float)) and cconf < 0.45:
            return True

        # plan expired + still uncertain
        with self._lock:
            plan_ts = self._last_plan_ts
        if plan_ts and (now - plan_ts) > self.cfg.plan_ttl_sec and cconf < 0.60:
            return True

        return False

    def _trigger_upload(self, roi_bgr: np.ndarray, dist_cm: Optional[float], local: Dict[str, Any]):
        self._last_trigger_ts = time.time()

        send = cv2.resize(roi_bgr, (self.cfg.send_w, self.cfg.send_h), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", send, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])
        if not ok:
            return

        # include local corridor context so GPT hiểu “lối đi”
        meta = {
            "ts": time.time(),
            "rotate180": self.rotate180,
            "lidar_cm": dist_cm,
            "lidar_strength": local.get("lidar_strength"),

            "roi_w": int(roi_bgr.shape[1]),
            "roi_h": int(roi_bgr.shape[0]),
            "send_w": self.cfg.send_w,
            "send_h": self.cfg.send_h,

            # local context hints for GPT
            "local_method": local.get("method"),
            "corridor_center_x": local.get("corridor_center_x"),
            "corridor_width_px": local.get("corridor_width_px"),
            "corridor_width_ratio": local.get("corridor_width_ratio"),
            "corridor_conf": local.get("corridor_conf"),
            "best_sector_local": local.get("best_sector"),
        }

        files = {
            "image": ("roi.jpg", jpg.tobytes(), "image/jpeg"),
            "meta": (None, json.dumps(meta), "application/json"),
        }

        try:
            r = self.http.post(self.cfg.server_url, files=files, timeout=self.cfg.gpt_timeout_sec)
            if r.ok:
                plan = r.json()
                with self._lock:
                    self._last_plan = plan
                    self._last_plan_ts = time.time()
        except Exception:
            pass
