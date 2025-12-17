#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple

import cv2
import numpy as np
import requests


@dataclass
class AvoidCfg:
    # camera
    cam_dev: str = "/dev/video0"
    cam_w: int = 640
    cam_h: int = 480
    cam_fps: int = 30

    # loop
    loop_hz: float = 15.0

    # ROI near-field (chỉ nhìn gần để tránh “rèm xa”)
    roi_y_start_ratio: float = 0.60  # lấy 40% phía dưới
    roi_y_end_ratio: float = 1.00

    # sector
    sector_n: int = 9

    # ultrasonic trigger
    trigger_cm: float = 120.0     # dưới ngưỡng này thì gọi server
    hard_stop_cm: float = 35.0    # cực gần thì dừng ngay (bạn tự dùng)
    min_valid_cm: float = 2.0
    max_valid_cm: float = 400.0

    # trigger debounce
    min_trigger_interval_sec: float = 2.0   # tránh spam server
    plan_ttl_sec: float = 8.0               # plan “hết hạn” sau vài giây

    # upload
    server_url: str = "https://YOUR_SERVER.up.railway.app/avoid_obstacle_vision"
    jpeg_quality: int = 55
    send_w: int = 256
    send_h: int = 144

    # local freespace (HSV)
    floor_sample_h: int = 16       # lấy 16px dưới cùng ROI làm “mẫu sàn”
    hsv_tolerance: Tuple[int, int, int] = (18, 60, 70)  # (H,S,V)


class AvoidObstacle:
    """
    Local loop: ultrasonic + camera ROI free-space
    Event trigger -> upload ROI image -> receive GPT plan
    Provide overlay drawing for WebDashboard.
    """

    def __init__(
        self,
        get_ultrasonic_cm: Callable[[], Optional[float]],
        cfg: AvoidCfg = AvoidCfg(),
        session: Optional[requests.Session] = None,
        get_frame_bgr: Optional[Callable[[], Any]] = None,

        # NEW: rotate camera 180 degrees (default ON)
        rotate180: bool = True,
    ):
        self.get_ultrasonic_cm = get_ultrasonic_cm
        self.cfg = cfg
        self.http = session or requests.Session()

        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self.get_frame_bgr = get_frame_bgr

        self.rotate180 = bool(rotate180)

        self._lock = threading.Lock()
        self._last_frame = None  # full frame BGR (after rotation, if enabled)
        self._last_roi = None    # roi BGR (after rotation)
        self._last_local = {}    # local scores
        self._last_plan: Dict[str, Any] = {}
        self._last_plan_ts = 0.0
        self._last_trigger_ts = 0.0

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
                "rotate180": self.rotate180,
            }

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Draw ROI + best sector + GPT plan overlay.
        NOTE: WebDashboard may rotate the frame too.
        Best practice:
          - Either rotate in AvoidObstacle loop and let WebDash NOT rotate,
          - OR rotate only in WebDash.
        You asked default rotate everywhere; to keep overlays matching,
        we rotate in AvoidObstacle loop, and WebDash rotates for display.
        (If you see overlay mismatch, set WebDash rotate180=False.)
        """
        if frame_bgr is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        y0 = int(self.cfg.roi_y_start_ratio * h)
        y1 = int(self.cfg.roi_y_end_ratio * h)

        # ROI boundary
        cv2.rectangle(frame_bgr, (0, y0), (w - 1, y1 - 1), (200, 200, 200), 1)
        cv2.putText(frame_bgr, "ROI near-field", (8, max(18, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        with self._lock:
            local = dict(self._last_local or {})
            plan = dict(self._last_plan or {})
            plan_ts = self._last_plan_ts

        # Local best sector highlight
        best = local.get("best_sector", None)
        if isinstance(best, int) and 0 <= best < self.cfg.sector_n:
            sw = w // self.cfg.sector_n
            sx0 = best * sw
            sx1 = (best + 1) * sw if best < self.cfg.sector_n - 1 else w
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (sx0, y0), (sx1, y1), (0, 255, 0), -1)
            frame_bgr[:] = cv2.addWeighted(overlay, 0.15, frame_bgr, 0.85, 0)

        # GPT overlay (ROI coords)
        now = time.time()
        if plan and plan_ts and (now - plan_ts) <= self.cfg.plan_ttl_sec:
            roi_w = w
            roi_h = y1 - y0

            safe_poly = plan.get("safe_poly", None)
            if isinstance(safe_poly, list) and len(safe_poly) >= 3:
                pts = []
                for p in safe_poly:
                    if isinstance(p, list) and len(p) == 2:
                        px, py = int(p[0]), int(p[1])
                        px = max(0, min(roi_w - 1, px))
                        py = max(0, min(roi_h - 1, py))
                        pts.append([px, py + y0])
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    overlay = frame_bgr.copy()
                    cv2.fillPoly(overlay, [pts_np], (0, 255, 0))
                    frame_bgr[:] = cv2.addWeighted(overlay, 0.18, frame_bgr, 0.82, 0)
                    cv2.polylines(frame_bgr, [pts_np], True, (0, 255, 0), 2)

            obstacles = plan.get("obstacles", [])
            if isinstance(obstacles, list):
                for ob in obstacles[:12]:
                    bbox = ob.get("bbox", None)  # [x1,y1,x2,y2] in ROI coords
                    label = str(ob.get("label", "obj"))
                    risk = ob.get("risk", None)
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x1, yb1, x2, yb2 = [int(v) for v in bbox]
                        x1 = max(0, min(roi_w - 1, x1))
                        x2 = max(0, min(roi_w - 1, x2))
                        yb1 = max(0, min(roi_h - 1, yb1))
                        yb2 = max(0, min(roi_h - 1, yb2))
                        cv2.rectangle(frame_bgr, (x1, y0 + yb1), (x2, y0 + yb2), (0, 200, 255), 2)
                        txt = f"{label}"
                        if isinstance(risk, (int, float)):
                            txt += f" r={risk:.2f}"
                        cv2.putText(frame_bgr, txt, (x1, y0 + max(12, yb1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

            nobs = plan.get("n_obstacles", None)
            best_sector = plan.get("best_sector", None)
            conf = plan.get("confidence", None)
            hud = f"GPT: obs={nobs} best={best_sector} conf={conf}"
            cv2.putText(frame_bgr, hud, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

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
                # NEW: rotate once here so EVERYTHING below uses correct orientation
                if self.rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                roi = self._extract_roi(frame)
                local = self._local_freespace_sector(roi)
                dist = self._read_ultra()

                local["ultra_cm"] = dist
                local["ts"] = time.time()

                with self._lock:
                    self._last_frame = frame
                    self._last_roi = roi
                    self._last_local = local

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

    def _read_ultra(self) -> Optional[float]:
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

    def _local_freespace_sector(self, roi_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        sample_h = min(self.cfg.floor_sample_h, h)
        sample = hsv[h - sample_h:h, :, :]
        med = np.median(sample.reshape(-1, 3), axis=0)
        mh, ms, mv = int(med[0]), int(med[1]), int(med[2])

        th, ts, tv = self.cfg.hsv_tolerance
        lo = np.array([max(0, mh - th), max(0, ms - ts), max(0, mv - tv)], dtype=np.uint8)
        hi = np.array([min(179, mh + th), min(255, ms + ts), min(255, mv + tv)], dtype=np.uint8)

        mask_floor = cv2.inRange(hsv, lo, hi)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_floor = cv2.morphologyEx(mask_floor, cv2.MORPH_OPEN, k, iterations=1)
        mask_floor = cv2.morphologyEx(mask_floor, cv2.MORPH_CLOSE, k, iterations=1)

        sector_n = self.cfg.sector_n
        sw = w // sector_n
        scores = []
        for i in range(sector_n):
            x0 = i * sw
            x1 = (i + 1) * sw if i < sector_n - 1 else w
            m = mask_floor[:, x0:x1]
            scores.append(float(np.mean(m > 0)))

        best = int(np.argmax(scores)) if scores else 0
        confidence = float(max(scores)) if scores else 0.0

        return {
            "scores": scores,
            "best_sector": best,
            "confidence": confidence,
            "floor_hsv_med": [mh, ms, mv],
        }

    def _should_trigger(self, ultra_cm: Optional[float], local: Dict[str, Any]) -> bool:
        now = time.time()
        if (now - self._last_trigger_ts) < self.cfg.min_trigger_interval_sec:
            return False

        if ultra_cm is not None and ultra_cm <= self.cfg.trigger_cm:
            return True

        conf = local.get("confidence", 0.0)
        if isinstance(conf, (int, float)) and conf < 0.35:
            return True

        with self._lock:
            plan_ts = self._last_plan_ts
        if plan_ts and (now - plan_ts) > self.cfg.plan_ttl_sec and conf < 0.55:
            return True

        return False

    def _trigger_upload(self, roi_bgr: np.ndarray, ultra_cm: Optional[float], local: Dict[str, Any]):
        self._last_trigger_ts = time.time()

        # IMPORTANT: You asked "rotate before send".
        # Here ROI is already rotated because frame rotated in _loop().
        # If later you disable rotate in _loop(), you can rotate here instead.
        roi_send = roi_bgr

        # resize + encode
        send = cv2.resize(roi_send, (self.cfg.send_w, self.cfg.send_h), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", send, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])
        if not ok:
            return

        meta = {
            "ultra_cm": ultra_cm,
            "local_best_sector": local.get("best_sector"),
            "local_scores": local.get("scores"),
            "ts": time.time(),
            "roi_w": int(roi_bgr.shape[1]),
            "roi_h": int(roi_bgr.shape[0]),
            "send_w": self.cfg.send_w,
            "send_h": self.cfg.send_h,
            "rotate180": self.rotate180,
        }

        files = {
            "image": ("roi.jpg", jpg.tobytes(), "image/jpeg"),
            "meta": (None, json.dumps(meta), "application/json"),
        }

        try:
            r = self.http.post(self.cfg.server_url, files=files, timeout=8)
            if r.ok:
                plan = r.json()
                with self._lock:
                    self._last_plan = plan
                    self._last_plan_ts = time.time()
        except Exception:
            pass
