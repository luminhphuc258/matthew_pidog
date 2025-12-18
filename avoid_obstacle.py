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

    roi_y_start_ratio: float = 0.55
    roi_y_end_ratio: float = 1.00

    sector_n: int = 9

    # LiDAR priority thresholds
    force_stop_cm: float = 40.0
    trigger_cm: float = 120.0
    hard_stop_cm: float = 28.0
    min_valid_cm: float = 2.0
    max_valid_cm: float = 800.0

    min_trigger_interval_sec: float = 1.2
    plan_ttl_sec: float = 6.0

    server_url: str = "https://embeddedprogramming-healtheworldserver.up.railway.app/avoid_obstacle_vision"
    jpeg_quality: int = 55
    send_w: int = 256
    send_h: int = 144

    # floor mask tuning
    floor_sample_h: int = 18
    hsv_tolerance: Tuple[int, int, int] = (18, 65, 75)

    # 6 bands (bottom is nearest)
    bands_n: int = 6
    floor_ok_ratio_band0: float = 0.25

    # corridor detect (from floor mask)
    corridor_floor_threshold: float = 0.45
    corridor_min_width_ratio: float = 0.18
    corridor_min_conf: float = 0.25

    narrow_replan_cooldown_sec: float = 2.5

    # plan->action selection
    min_plan_confidence: float = 0.40
    min_local_confidence: float = 0.25


class AvoidObstacle:
    """
    - Local: estimate floor free corridor using HSV floor mask + 6 bottom bands.
    - Trigger (distance near / floor blocked / low conf / forced): upload ROI to server (GPT vision)
    - Plan: obstacles + walkway polygon + best_sector (+ optional action)
    - Provide: draw_overlay() and get_best_action() for main loop.
    """

    def __init__(
        self,
        get_distance_cm: Optional[Callable[[], Optional[float]]] = None,
        cfg: AvoidCfg = AvoidCfg(),
        session: Optional[requests.Session] = None,
        get_frame_bgr: Optional[Callable[[], Any]] = None,
        rotate180: bool = True,
        get_lidar_strength: Optional[Callable[[], Optional[float]]] = None,

        # compatibility: older call sites
        get_ultrasonic_cm: Optional[Callable[[], Optional[float]]] = None,
        **kwargs,
    ):
        # allow passing getter in multiple names
        self.get_distance_cm = get_distance_cm or get_ultrasonic_cm
        if self.get_distance_cm is None:
            # also accept alternative names via kwargs
            for k in ("get_lidar_cm", "get_dist_cm", "get_distance"):
                if k in kwargs and kwargs[k] is not None:
                    self.get_distance_cm = kwargs[k]
                    break
        if self.get_distance_cm is None:
            raise ValueError("AvoidObstacle missing distance getter (get_distance_cm or get_ultrasonic_cm).")

        self.get_lidar_strength = get_lidar_strength
        self.cfg = cfg
        self.http = session or requests.Session()

        self.get_frame_bgr = get_frame_bgr
        self.rotate180 = bool(rotate180)

        self._cap = None
        self._thread = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._last_frame = None
        self._last_roi = None

        self._last_local: Dict[str, Any] = {}
        self._last_plan: Dict[str, Any] = {}
        self._last_plan_ts = 0.0

        self._last_trigger_ts = 0.0
        self._last_force_ts = 0.0

        self._upload_lock = threading.Lock()
        self._upload_inflight = False

        self._last_narrow_replan_ts = 0.0

    # ---------------- public API ----------------

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
            age = (now - self._last_plan_ts) if self._last_plan_ts else None
            return {
                "local": dict(self._last_local or {}),
                "plan": dict(self._last_plan or {}),
                "plan_age_sec": age,
                "rotate180": self.rotate180,
                "upload_inflight": bool(self._upload_inflight),
            }

    def force_trigger(self, reason: str = "force_trigger") -> bool:
        """Force upload latest ROI right now (async)."""
        return self.request_plan_now(reason=reason, force=True)

    def request_plan_now(self, reason: str = "manual_force", force: bool = True) -> bool:
        with self._lock:
            roi = None if self._last_roi is None else self._last_roi.copy()
            local = dict(self._last_local or {})
        if roi is None:
            return False
        dist = local.get("dist_cm", None)
        local["force_reason"] = reason
        return self._trigger_upload_async(roi, dist, local, force=force)

    def plan_is_fresh(self) -> bool:
        with self._lock:
            ts = float(self._last_plan_ts or 0.0)
        if ts <= 0:
            return False
        return (time.time() - ts) <= self.cfg.plan_ttl_sec

    def get_best_action(self) -> Optional[Dict[str, Any]]:
        """
        Return dict with:
          {action, confidence, best_sector, obstacles, walkway_poly, no_path, narrow, walkway_width_ratio}
        """
        with self._lock:
            local = dict(self._last_local or {})
            plan = dict(self._last_plan or {})
            plan_ts = float(self._last_plan_ts or 0.0)

        now = time.time()
        plan_ok = bool(plan) and plan_ts and (now - plan_ts) <= self.cfg.plan_ttl_sec

        # local corridor fallback
        corridor_poly = local.get("corridor_poly")
        corridor_conf = float(local.get("corridor_conf", 0.0) or 0.0)
        corridor_w = float(local.get("corridor_width_ratio", 0.0) or 0.0)
        floor_blocked = bool(local.get("floor_blocked", False))
        local_best_sector = local.get("best_sector", None)

        out: Dict[str, Any] = {}
        if plan_ok:
            out.update(plan)

        # Normalize walkway poly
        if "walkway_poly" not in out:
            if isinstance(out.get("safe_poly"), list):
                out["walkway_poly"] = out.get("safe_poly")
            else:
                out["walkway_poly"] = corridor_poly

        # Normalize confidence
        if "confidence" not in out or not isinstance(out.get("confidence"), (int, float)):
            out["confidence"] = corridor_conf

        # Normalize best_sector
        if "best_sector" not in out or not isinstance(out.get("best_sector"), int):
            if isinstance(local_best_sector, int):
                out["best_sector"] = int(local_best_sector)
            else:
                out["best_sector"] = self.cfg.sector_n // 2

        # Estimate walkway width ratio if missing
        if "walkway_width_ratio" not in out or not isinstance(out.get("walkway_width_ratio"), (int, float)):
            out["walkway_width_ratio"] = corridor_w

        # Determine narrow/no_path
        wwr = float(out.get("walkway_width_ratio", 0.0) or 0.0)
        conf = float(out.get("confidence", 0.0) or 0.0)

        narrow = False
        no_path = False
        if floor_blocked:
            no_path = True
        if wwr > 0 and wwr < self.cfg.corridor_min_width_ratio and conf >= self.cfg.corridor_min_conf:
            narrow = True
        if wwr <= 0.10 and conf >= 0.15:
            no_path = True

        out["narrow"] = bool(out.get("narrow", False) or narrow)
        out["no_path"] = bool(out.get("no_path", False) or no_path)

        # Decide action if not provided by server:
        # - if no_path => STOP
        # - else turn towards best_sector if not center-ish
        # - else FORWARD
        if not isinstance(out.get("action"), str):
            bs = int(out.get("best_sector", self.cfg.sector_n // 2))
            center = self.cfg.sector_n // 2
            if out["no_path"]:
                out["action"] = "STOP"
            else:
                if bs <= center - 1:
                    out["action"] = "LEFT"
                elif bs >= center + 1:
                    out["action"] = "RIGHT"
                else:
                    out["action"] = "FORWARD"

        # If confidence too low => return minimal (caller can fall back to planner)
        if float(out.get("confidence", 0.0) or 0.0) < min(self.cfg.min_plan_confidence, 0.25) and not plan_ok:
            return None

        # keep obstacles as list
        if not isinstance(out.get("obstacles"), list):
            out["obstacles"] = []

        return out

    def should_rotate_replan(self) -> bool:
        """If corridor too narrow or floor blocked, suggest rotate 180 + replan (cooldown)."""
        now = time.time()
        with self._lock:
            local = dict(self._last_local or {})
            plan = dict(self._last_plan or {})

        if (now - self._last_narrow_replan_ts) < self.cfg.narrow_replan_cooldown_sec:
            return False

        cw = local.get("corridor_width_ratio", None)
        cc = local.get("corridor_conf", None)
        floor_blocked = bool(local.get("floor_blocked", False))

        bad = False
        if floor_blocked:
            bad = True

        if isinstance(cw, (int, float)) and isinstance(cc, (int, float)):
            if float(cw) < self.cfg.corridor_min_width_ratio and float(cc) >= self.cfg.corridor_min_conf:
                bad = True

        pw = plan.get("walkway_width_ratio", None)
        if isinstance(pw, (int, float)) and float(pw) < self.cfg.corridor_min_width_ratio:
            bad = True

        if bad:
            self._last_narrow_replan_ts = now
            return True
        return False

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Overlay for WebDashboard: walkway yellow + obstacles red + short HUD."""
        if frame_bgr is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        y0 = int(self.cfg.roi_y_start_ratio * h)
        y1 = int(self.cfg.roi_y_end_ratio * h)
        roi_h = max(1, y1 - y0)

        with self._lock:
            local = dict(self._last_local or {})
            plan = dict(self._last_plan or {})
            plan_ts = float(self._last_plan_ts or 0.0)

        now = time.time()
        plan_ok = bool(plan) and plan_ts and (now - plan_ts) <= self.cfg.plan_ttl_sec

        # walkway poly: prefer GPT plan, fallback local corridor
        walkway_poly = None
        if plan_ok and isinstance(plan.get("walkway_poly", None), list):
            walkway_poly = plan.get("walkway_poly")
        elif plan_ok and isinstance(plan.get("safe_poly", None), list):
            walkway_poly = plan.get("safe_poly")
        else:
            walkway_poly = local.get("corridor_poly", None)

        if isinstance(walkway_poly, list) and len(walkway_poly) >= 3:
            pts = []
            for p in walkway_poly:
                if isinstance(p, list) and len(p) == 2:
                    px, py = int(p[0]), int(p[1])
                    px = max(0, min(w - 1, px))
                    py = max(0, min(roi_h - 1, py))
                    pts.append([px, py + y0])
            if len(pts) >= 3:
                pts_np = np.array(pts, dtype=np.int32)
                overlay = frame_bgr.copy()
                cv2.fillPoly(overlay, [pts_np], (0, 255, 255))  # yellow
                frame_bgr[:] = cv2.addWeighted(overlay, 0.22, frame_bgr, 0.78, 0)
                cv2.polylines(frame_bgr, [pts_np], True, (0, 255, 255), 2)

        # obstacles only from plan (if fresh)
        obstacles = plan.get("obstacles", []) if plan_ok else []
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
                    cv2.rectangle(frame_bgr, (x1, y0 + yb1), (x2, y0 + yb2), (0, 0, 255), 2)
                    txt = label
                    if isinstance(risk, (int, float)):
                        txt += f" {float(risk):.2f}"
                    cv2.putText(
                        frame_bgr,
                        txt,
                        (x1, y0 + max(14, yb1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

        dist = local.get("dist_cm", None)
        conf = plan.get("confidence", None) if plan_ok else None
        if conf is None:
            conf = local.get("corridor_conf", None)
        best = plan.get("best_sector", None) if plan_ok else local.get("best_sector", None)
        floor_blocked = local.get("floor_blocked", False)

        hud = f"d={dist}cm  best={best}  conf={conf}  blocked={int(bool(floor_blocked))}"
        cv2.putText(frame_bgr, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return frame_bgr

    # ---------------- internal loop ----------------

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
                    ok = False
                    frame = None
            else:
                ok, frame = self._cap.read() if self._cap else (False, None)

            if ok and frame is not None:
                if self.rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                roi = self._extract_roi(frame)
                dist = self._read_distance_cm()
                strength = self._read_strength()

                local = self._analyze_floor_and_corridor(roi)
                local["dist_cm"] = dist
                local["lidar_strength"] = strength
                local["ts"] = time.time()

                with self._lock:
                    self._last_frame = frame
                    self._last_roi = roi
                    self._last_local = local

                force = bool(dist is not None and dist <= self.cfg.force_stop_cm)
                if self._should_trigger(dist, local, force=force):
                    self._trigger_upload_async(roi, dist, local, force=force)

            sleep_s = dt - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _extract_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        y0 = int(self.cfg.roi_y_start_ratio * h)
        y1 = int(self.cfg.roi_y_end_ratio * h)
        return frame_bgr[y0:y1, 0:w].copy()

    def _read_distance_cm(self) -> Optional[float]:
        try:
            d = self.get_distance_cm()
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

    # ---------------- floor + corridor analysis ----------------

    def _floor_mask(self, roi_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        sample_h = min(self.cfg.floor_sample_h, h)
        sample = hsv[h - sample_h:h, :, :]
        med = np.median(sample.reshape(-1, 3), axis=0)
        mh, ms, mv = int(med[0]), int(med[1]), int(med[2])

        th, ts, tv = self.cfg.hsv_tolerance
        lo = np.array([max(0, mh - th), max(0, ms - ts), max(0, mv - tv)], dtype=np.uint8)
        hi = np.array([min(179, mh + th), min(255, ms + ts), min(255, mv + tv)], dtype=np.uint8)

        mask = cv2.inRange(hsv, lo, hi)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        dbg = {"floor_hsv_med": [mh, ms, mv]}
        return mask, dbg

    def _analyze_floor_and_corridor(self, roi_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = roi_bgr.shape[:2]
        mask_floor, dbg = self._floor_mask(roi_bgr)

        bands_n = max(2, int(self.cfg.bands_n))
        band_h = max(1, h // bands_n)

        band_ratios: List[float] = []
        for i in range(bands_n):
            y0 = h - (i + 1) * band_h
            y1 = h - i * band_h if i < bands_n - 1 else h
            y0 = max(0, y0)
            y1 = min(h, y1)
            m = mask_floor[y0:y1, :]
            band_ratios.append(float(np.mean(m > 0)))

        floor_blocked = band_ratios[0] < self.cfg.floor_ok_ratio_band0

        # corridor from lower half: find widest good-floor segment
        y_cut = int(0.45 * h)
        m2 = mask_floor[y_cut:h, :]
        col_ratio = np.mean(m2 > 0, axis=0)

        good = col_ratio >= self.cfg.corridor_floor_threshold
        best_len = 0
        best_l = 0
        best_r = 0
        cur_l = None
        for x in range(w):
            if good[x] and cur_l is None:
                cur_l = x
            if (not good[x] or x == w - 1) and cur_l is not None:
                cur_r = x if not good[x] else x + 1
                seg_len = cur_r - cur_l
                if seg_len > best_len:
                    best_len = seg_len
                    best_l, best_r = cur_l, cur_r
                cur_l = None

        corridor_width = best_len
        corridor_width_ratio = float(corridor_width) / float(max(1, w))
        corridor_center_x = int((best_l + best_r) / 2) if corridor_width > 0 else int(w / 2)

        conf = 0.0
        if corridor_width > 0:
            conf = 0.55 * min(1.0, corridor_width_ratio / 0.45) + 0.45 * min(1.0, band_ratios[0] / 0.70)

        sector_n = max(1, int(self.cfg.sector_n))
        sw = max(1, w // sector_n)
        best_sector = int(np.clip(corridor_center_x // sw, 0, sector_n - 1))

        if corridor_width > 0:
            x1 = int(best_l)
            x2 = int(best_r)
            shrink = int(0.20 * corridor_width)
            x1u = int(np.clip(x1 + shrink, 0, w - 1))
            x2u = int(np.clip(x2 - shrink, 0, w - 1))
            yb = h - 1
            yt = int(0.52 * h)
            corridor_poly = [[x1, yb], [x2, yb], [x2u, yt], [x1u, yt]]
        else:
            halfw = int(0.18 * w)
            x1 = max(0, corridor_center_x - halfw)
            x2 = min(w - 1, corridor_center_x + halfw)
            corridor_poly = [[x1, h - 1], [x2, h - 1], [x2, int(0.55 * h)], [x1, int(0.55 * h)]]

        return {
            "band_floor_ratios": band_ratios,
            "floor_blocked": bool(floor_blocked),
            "corridor_center_x": corridor_center_x,
            "corridor_width_ratio": float(corridor_width_ratio),
            "corridor_conf": float(conf),
            "best_sector": int(best_sector),
            "corridor_poly": corridor_poly,
            **dbg,
        }

    # ---------------- trigger/upload ----------------

    def _should_trigger(self, dist_cm: Optional[float], local: Dict[str, Any], force: bool = False) -> bool:
        now = time.time()

        if not force and (now - self._last_trigger_ts) < self.cfg.min_trigger_interval_sec:
            return False

        if force:
            if (now - self._last_force_ts) < 0.7:
                return False
            return True

        if dist_cm is not None and dist_cm <= self.cfg.trigger_cm:
            return True

        if bool(local.get("floor_blocked", False)):
            return True

        conf = local.get("corridor_conf", 0.0)
        if isinstance(conf, (int, float)) and float(conf) < 0.20:
            return True

        with self._lock:
            plan_ts = self._last_plan_ts
        if plan_ts and (now - plan_ts) > self.cfg.plan_ttl_sec and float(conf) < 0.45:
            return True

        return False

    def _trigger_upload_async(self, roi_bgr: np.ndarray, dist_cm: Optional[float], local: Dict[str, Any], force: bool) -> bool:
        with self._upload_lock:
            if self._upload_inflight:
                return True
            self._upload_inflight = True

        if force:
            self._last_force_ts = time.time()
        self._last_trigger_ts = time.time()

        def worker():
            try:
                self._trigger_upload(roi_bgr, dist_cm, local, force=force)
            finally:
                with self._upload_lock:
                    self._upload_inflight = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _trigger_upload(self, roi_bgr: np.ndarray, dist_cm: Optional[float], local: Dict[str, Any], force: bool):
        send = cv2.resize(roi_bgr, (self.cfg.send_w, self.cfg.send_h), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", send, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])
        if not ok:
            return

        meta = {
            "lidar_cm": dist_cm,
            "lidar_strength": local.get("lidar_strength"),
            "best_sector_local": local.get("best_sector"),
            "corridor_center_x": local.get("corridor_center_x"),
            "corridor_width_ratio": local.get("corridor_width_ratio"),
            "corridor_conf": local.get("corridor_conf"),
            "floor_blocked": local.get("floor_blocked"),
            "band_floor_ratios": local.get("band_floor_ratios"),
            "ts": time.time(),
            "roi_w": int(roi_bgr.shape[1]),
            "roi_h": int(roi_bgr.shape[0]),
            "send_w": int(self.cfg.send_w),
            "send_h": int(self.cfg.send_h),
            "rotate180": bool(self.rotate180),
            "force": bool(force),
            "force_reason": local.get("force_reason", None),
        }

        files = {
            "image": ("roi.jpg", jpg.tobytes(), "image/jpeg"),
            "meta": (None, json.dumps(meta), "application/json"),
        }

        try:
            r = self.http.post(self.cfg.server_url, files=files, timeout=8)
            if not r.ok:
                return

            plan = r.json()
            if not isinstance(plan, dict):
                return

            # normalize: walkway_poly
            if "walkway_poly" not in plan and "safe_poly" in plan:
                plan["walkway_poly"] = plan.get("safe_poly")

            # walkway_width_ratio
            if "walkway_width_ratio" not in plan:
                wp = plan.get("walkway_poly", None)
                if isinstance(wp, list) and len(wp) >= 2:
                    try:
                        xs = [float(p[0]) for p in wp if isinstance(p, list) and len(p) == 2]
                        if xs:
                            wroi = max(1.0, float(roi_bgr.shape[1]))
                            plan["walkway_width_ratio"] = float((max(xs) - min(xs)) / wroi)
                    except Exception:
                        pass

            # n_obstacles
            if "n_obstacles" not in plan:
                obs = plan.get("obstacles", [])
                plan["n_obstacles"] = len(obs) if isinstance(obs, list) else 0

            with self._lock:
                self._last_plan = plan
                self._last_plan_ts = time.time()
        except Exception:
            pass
