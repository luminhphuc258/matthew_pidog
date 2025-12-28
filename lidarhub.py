#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LiDAR Hub (Port 9399)
- Endpoints:
    GET  /api/status              -> JSON status
    GET  /api/decision_label      -> TEXT: STOP / GO_STRAIGHT / TURN_LEFT / TURN_RIGHT
    GET  /api/decision            -> JSON decision + sectors (debug)
    GET  /take_lidar_data         -> JSON last scan (points + derived fields)
    POST /ingest_scan             -> PUSH raw scan into hub (recommended)
    POST /api/pose                -> Optional: update pose from your odom (x,y,yaw_deg)
- Decision recomputed continuously (no sticky direction).
"""

import time
import math
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request, Response

# =========================
# CONFIG
# =========================

HOST = "0.0.0.0"
PORT = 9399

# How often to recompute decision (Hz). User asked "mỗi giây" -> set to 1.0 if you want.
DECISION_HZ = 5.0  # change to 1.0 if you want exactly once per second

# Scan staleness: if last scan older than this -> STOP fail-safe
MAX_SCAN_AGE_S = 0.50

# Mode to get scans:
#   "PUSH"      -> your LiDAR reader POSTs scans to /ingest_scan
#   "PULL_HTTP" -> hub pulls from RAW_SCAN_URL
SCAN_MODE = "PUSH"

# If PULL_HTTP:
RAW_SCAN_URL = "http://127.0.0.1:9398/scan"  # change if you have a raw scan server
RAW_SCAN_TIMEOUT_S = 0.25
RAW_SCAN_PULL_HZ = 10.0

# Sector geometry (k=3): CENTER, LEFT, RIGHT
CENTER_HALF_DEG = 28.0     # CENTER is [-28..+28] degrees relative to front
FRONT_ARC_DEG = 85.0       # "front hemisphere" considered for safety (wider than CENTER)

# Filtering
MIN_QUALITY = 5            # ignore points with q < MIN_QUALITY
MAX_RANGE_M = 12.0
USE_ONLY_FRONT_180 = False  # If True -> only consider rel_deg in [-90..+90] for decision

# Safety distances (tune!)
HARD_STOP_M = 0.35         # if any obstacle in FRONT_ARC within this -> STOP
TURN_REQUIRED_M = 0.75     # if obstacle in FRONT_ARC within this -> must TURN (not GO_STRAIGHT)
GO_CLEAR_M = 0.90          # need at least this clearance to GO_STRAIGHT comfortably

# If one side is too close, bias to turn away even if center looks ok
SIDE_TOO_CLOSE_M = 0.55

# Optional: mirror left/right if your coordinate is flipped
MIRROR_LEFT_RIGHT = False

# Small helper: clamp
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _now() -> float:
    return time.time()

def _ang_norm_360(deg: float) -> float:
    deg = deg % 360.0
    return deg + 360.0 if deg < 0 else deg

def _ang_norm_180(deg: float) -> float:
    """Normalize to [-180, +180]."""
    deg = (deg + 180.0) % 360.0 - 180.0
    return deg

def _safe_float(x: float) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return float(x)

@dataclass
class SectorStats:
    count: int = 0
    min_dist: Optional[float] = None
    avg_dist: Optional[float] = None

@dataclass
class Decision:
    label: str = "STOP"
    reason: str = "boot"
    ts: float = 0.0
    sectors: Dict[str, SectorStats] = None
    front: Dict[str, Any] = None

class LidarHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}  # optional external updates

        self._front_center_deg = 180.0  # default; update via ingest if provided
        self._mirror = bool(MIRROR_LEFT_RIGHT)

        self._last_scan: Optional[Dict[str, Any]] = None
        self._last_scan_ts: float = 0.0

        self._decision = Decision(
            label="STOP",
            reason="boot",
            ts=_now(),
            sectors={"LEFT": SectorStats(), "CENTER": SectorStats(), "RIGHT": SectorStats()},
            front={"min_dist": None, "min_rel_deg": None, "min_angle": None}
        )

        self._running = True
        self._pull_thread = None
        self._decision_thread = None

    # -------------------------
    # External interface
    # -------------------------
    def update_pose(self, x: float, y: float, yaw_deg: float) -> None:
        with self._lock:
            self._pose = {"x": float(x), "y": float(y), "yaw_deg": float(yaw_deg)}

    def ingest_scan(self, scan: Dict[str, Any]) -> None:
        """
        scan expected minimal format:
        {
          "ts": optional timestamp,
          "frame": { "front_center_deg_used": ..., "back_deg_est": ..., "max_range_m": ... } optional,
          "points": [ {"angle": deg, "dist_m": float, "q": int} ... ]  (dist_cm also accepted)
        }
        """
        ts = float(scan.get("ts") or _now())

        frame = scan.get("frame") or {}
        # Prefer explicit front center; else compute from back_deg_est if present
        if "front_center_deg_used" in frame:
            fcd = float(frame["front_center_deg_used"])
        elif "front_center_deg" in frame:
            fcd = float(frame["front_center_deg"])
        elif "back_deg_est" in frame:
            fcd = _ang_norm_360(float(frame["back_deg_est"]) + 180.0)
        else:
            fcd = None

        points = scan.get("points") or []
        # Normalize points: angle, dist_m, q
        norm_points: List[Dict[str, Any]] = []
        for p in points:
            try:
                ang = float(p.get("angle", p.get("theta", 0.0)))
                q = int(p.get("q", p.get("quality", 0)))
                if "dist_m" in p:
                    dist_m = float(p["dist_m"])
                elif "dist_cm" in p:
                    dist_m = float(p["dist_cm"]) / 100.0
                else:
                    continue
                if dist_m <= 0:
                    continue
                norm_points.append({"angle": ang, "dist_m": dist_m, "q": q})
            except Exception:
                continue

        with self._lock:
            if fcd is not None:
                self._front_center_deg = _ang_norm_360(fcd)
            # max range (optional)
            if "max_range_m" in frame:
                # keep max_range if needed later
                pass

            self._last_scan = {
                "ok": True,
                "ts": ts,
                "n": len(norm_points),
                "frame": {
                    "front_center_deg_used": self._front_center_deg,
                    "mirror": int(self._mirror),
                    "max_range_m": float(frame.get("max_range_m", MAX_RANGE_M)),
                },
                "points_raw": norm_points,
            }
            self._last_scan_ts = ts

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            scan_age = _now() - self._last_scan_ts if self._last_scan_ts else None
            dec_age = _now() - self._decision.ts if self._decision and self._decision.ts else None
            return {
                "running": bool(self._running),
                "scan_age_s": _safe_float(scan_age),
                "decision_age_s": _safe_float(dec_age),
                "pose": dict(self._pose),
                "front_center": float(self._front_center_deg),
            }

    def get_decision_label(self) -> str:
        with self._lock:
            return str(self._decision.label)

    def get_decision_json(self) -> Dict[str, Any]:
        with self._lock:
            d = self._decision
            return {
                "label": d.label,
                "reason": d.reason,
                "ts": d.ts,
                "sectors": {
                    k: {
                        "count": int(v.count),
                        "min_dist": _safe_float(v.min_dist),
                        "avg_dist": _safe_float(v.avg_dist),
                    } for k, v in (d.sectors or {}).items()
                },
                "front": dict(d.front or {}),
            }

    def get_last_scan_debug(self) -> Dict[str, Any]:
        """
        Return last scan with derived fields similar to your current output:
        points: [{angle, dist_m, dist_cm, q, rel_deg, angle_front_360, is_front_180, k3_sector, x, y, ts}, ...]
        """
        with self._lock:
            scan = self._last_scan
            fcd = self._front_center_deg
            mirror = self._mirror
            pose = dict(self._pose)

        if not scan:
            return {"ok": False, "error": "no_scan"}

        pts = scan.get("points_raw", [])
        out_points: List[Dict[str, Any]] = []

        for p in pts:
            ang = float(p["angle"])
            dist_m = float(p["dist_m"])
            q = int(p["q"])

            # compute rel_deg: 0 means "front"
            rel = _ang_norm_180(_ang_norm_360(ang - fcd))

            # optional mirror
            if mirror:
                rel = -rel

            is_front_180 = (abs(rel) <= 90.0)

            # sector (k=3)
            k3 = None
            if is_front_180:
                if abs(rel) <= CENTER_HALF_DEG:
                    k3 = "CENTER"
                elif rel > CENTER_HALF_DEG:
                    k3 = "LEFT"
                else:
                    k3 = "RIGHT"

            # make angle_front_360 similar concept: 0..360 where 180 is "front_center"? (your client already handles)
            angle_front_360 = _ang_norm_360(ang - (fcd - 180.0))
            rel_deg = rel

            # coords for map (keep your "forward is negative x" style)
            rel_rad = math.radians(rel_deg)
            x = -dist_m * math.cos(rel_rad)
            y = -dist_m * math.sin(rel_rad)

            out_points.append({
                "angle": ang,
                "theta": ang,
                "dist_m": dist_m,
                "dist_cm": dist_m * 100.0,
                "q": q,
                "rel_deg": rel_deg,
                "angle_front_360": angle_front_360,
                "is_front_180": bool(is_front_180),
                "k3_sector": k3,
                "x": x,
                "y": y,
                "ts": float(scan.get("ts", _now())),
            })

        return {
            "ok": True,
            "ts": float(scan.get("ts", _now())),
            "last_point_ts": float(scan.get("ts", _now())),
            "n": len(out_points),
            "frame": {
                "front_center_deg_used": float(fcd),
                "mirror": int(mirror),
                "max_range_m": float(scan.get("frame", {}).get("max_range_m", MAX_RANGE_M)),
                "back_deg_est": float(_ang_norm_360(fcd - 180.0)),
            },
            "pose": pose,
            "points": out_points
        }

    # -------------------------
    # Threads
    # -------------------------
    def start(self) -> None:
        self._running = True

        if SCAN_MODE.upper() == "PULL_HTTP":
            self._pull_thread = threading.Thread(target=self._pull_loop, daemon=True)
            self._pull_thread.start()

        self._decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self._decision_thread.start()

    def _pull_loop(self) -> None:
        interval = 1.0 / max(1e-3, float(RAW_SCAN_PULL_HZ))
        while self._running:
            t0 = _now()
            try:
                r = requests.get(RAW_SCAN_URL, timeout=RAW_SCAN_TIMEOUT_S)
                if r.ok:
                    data = r.json()
                    # data should contain points; we accept flexible format
                    self.ingest_scan(data)
            except Exception:
                pass

            dt = _now() - t0
            time.sleep(max(0.0, interval - dt))

    def _decision_loop(self) -> None:
        interval = 1.0 / max(1e-3, float(DECISION_HZ))
        while self._running:
            t0 = _now()
            decision = self._compute_decision()
            with self._lock:
                self._decision = decision
            dt = _now() - t0
            time.sleep(max(0.0, interval - dt))

    # -------------------------
    # Core logic (FIXED)
    # -------------------------
    def _compute_decision(self) -> Decision:
        with self._lock:
            scan = self._last_scan
            scan_ts = self._last_scan_ts
            fcd = self._front_center_deg
            mirror = self._mirror

        now = _now()
        age = now - scan_ts if scan_ts else 999.0

        # Fail-safe
        if (not scan) or (age > MAX_SCAN_AGE_S):
            return Decision(
                label="STOP",
                reason=f"fail_safe_no_fresh_scan age={age:.3f}s",
                ts=now,
                sectors={"LEFT": SectorStats(), "CENTER": SectorStats(), "RIGHT": SectorStats()},
                front={"min_dist": None, "min_rel_deg": None, "min_angle": None}
            )

        pts = scan.get("points_raw", [])
        # Build derived list for decision quickly
        derived: List[Tuple[float, float, int]] = []  # (rel_deg, dist_m, q)
        for p in pts:
            q = int(p["q"])
            if q < MIN_QUALITY:
                continue
            dist_m = float(p["dist_m"])
            if dist_m <= 0 or dist_m > MAX_RANGE_M:
                continue

            ang = float(p["angle"])
            rel = _ang_norm_180(_ang_norm_360(ang - fcd))
            if mirror:
                rel = -rel

            # optionally only consider front half for decision
            if USE_ONLY_FRONT_180 and abs(rel) > 90.0:
                continue

            derived.append((rel, dist_m, q))

        if len(derived) < 30:
            # too few points -> fail-safe
            return Decision(
                label="STOP",
                reason=f"fail_safe_low_points n={len(derived)}",
                ts=now,
                sectors={"LEFT": SectorStats(), "CENTER": SectorStats(), "RIGHT": SectorStats()},
                front={"min_dist": None, "min_rel_deg": None, "min_angle": None}
            )

        # Compute FRONT ARC minimum (THIS is the key fix)
        front_min = None
        front_min_rel = None
        for rel, dist, _q in derived:
            if abs(rel) <= FRONT_ARC_DEG:
                if (front_min is None) or (dist < front_min):
                    front_min = dist
                    front_min_rel = rel

        # Sector stats in front-180 (for steering)
        def sector_name(rel_deg: float) -> Optional[str]:
            if abs(rel_deg) > 90.0:
                return None
            if abs(rel_deg) <= CENTER_HALF_DEG:
                return "CENTER"
            return "LEFT" if rel_deg > 0 else "RIGHT"

        buckets = {"LEFT": [], "CENTER": [], "RIGHT": []}
        for rel, dist, _q in derived:
            s = sector_name(rel)
            if s:
                buckets[s].append((rel, dist))

        def stats(vals: List[Tuple[float, float]]) -> SectorStats:
            if not vals:
                return SectorStats(count=0, min_dist=None, avg_dist=None)
            dists = [v[1] for v in vals]
            return SectorStats(
                count=len(dists),
                min_dist=min(dists),
                avg_dist=sum(dists) / max(1, len(dists))
            )

        s_left = stats(buckets["LEFT"])
        s_center = stats(buckets["CENTER"])
        s_right = stats(buckets["RIGHT"])

        # Helper: pick turn direction based on clearance
        left_min = s_left.min_dist if s_left.min_dist is not None else 999.0
        right_min = s_right.min_dist if s_right.min_dist is not None else 999.0
        center_min = s_center.min_dist if s_center.min_dist is not None else 999.0

        # Decision rules (no sticky)
        # 1) Hard stop if something extremely close in the front arc
        if (front_min is not None) and (front_min <= HARD_STOP_M):
            return Decision(
                label="STOP",
                reason=f"hard_stop front_min={front_min:.3f}m rel={front_min_rel:.1f}deg",
                ts=now,
                sectors={"LEFT": s_left, "CENTER": s_center, "RIGHT": s_right},
                front={"min_dist": front_min, "min_rel_deg": front_min_rel, "min_angle": None}
            )

        # 2) If something near in front arc -> must turn away (NOT go straight)
        if (front_min is not None) and (front_min <= TURN_REQUIRED_M):
            # turn to the side with bigger clearance
            if right_min >= left_min:
                lbl = "TURN_RIGHT"
                why = f"front_blocked front_min={front_min:.3f} -> turn_right (R={right_min:.3f} >= L={left_min:.3f})"
            else:
                lbl = "TURN_LEFT"
                why = f"front_blocked front_min={front_min:.3f} -> turn_left (L={left_min:.3f} > R={right_min:.3f})"

            return Decision(
                label=lbl,
                reason=why,
                ts=now,
                sectors={"LEFT": s_left, "CENTER": s_center, "RIGHT": s_right},
                front={"min_dist": front_min, "min_rel_deg": front_min_rel, "min_angle": None}
            )

        # 3) Even if center looks ok, if one side is too close, bias away (prevents body clipping barrel)
        if left_min <= SIDE_TOO_CLOSE_M and right_min > left_min:
            return Decision(
                label="TURN_RIGHT",
                reason=f"side_too_close_left L={left_min:.3f} -> turn_right",
                ts=now,
                sectors={"LEFT": s_left, "CENTER": s_center, "RIGHT": s_right},
                front={"min_dist": front_min, "min_rel_deg": front_min_rel, "min_angle": None}
            )
        if right_min <= SIDE_TOO_CLOSE_M and left_min > right_min:
            return Decision(
                label="TURN_LEFT",
                reason=f"side_too_close_right R={right_min:.3f} -> turn_left",
                ts=now,
                sectors={"LEFT": s_left, "CENTER": s_center, "RIGHT": s_right},
                front={"min_dist": front_min, "min_rel_deg": front_min_rel, "min_angle": None}
            )

        # 4) If center corridor not clear enough -> turn to better side
        if center_min < GO_CLEAR_M:
            if right_min >= left_min:
                lbl = "TURN_RIGHT"
                why = f"center_not_clear C={center_min:.3f} < {GO_CLEAR_M:.2f} -> turn_right"
            else:
                lbl = "TURN_LEFT"
                why = f"center_not_clear C={center_min:.3f} < {GO_CLEAR_M:.2f} -> turn_left"
            return Decision(
                label=lbl,
                reason=why,
                ts=now,
                sectors={"LEFT": s_left, "CENTER": s_center, "RIGHT": s_right},
                front={"min_dist": front_min, "min_rel_deg": front_min_rel, "min_angle": None}
            )

        # 5) Otherwise go straight
        return Decision(
            label="GO_STRAIGHT",
            reason=f"clear front_min={front_min:.3f} C={center_min:.3f}",
            ts=now,
            sectors={"LEFT": s_left, "CENTER": s_center, "RIGHT": s_right},
            front={"min_dist": front_min, "min_rel_deg": front_min_rel, "min_angle": None}
        )


# =========================
# Flask app
# =========================

app = Flask(__name__)
hub = LidarHub()
hub.start()

@app.get("/api/status")
def api_status():
    return jsonify(hub.get_status())

@app.get("/api/decision_label")
def api_decision_label():
    # Return plain text label
    return Response(hub.get_decision_label(), mimetype="text/plain")

@app.get("/api/decision")
def api_decision():
    return jsonify(hub.get_decision_json())

@app.get("/take_lidar_data")
def take_lidar_data():
    return jsonify(hub.get_last_scan_debug())

@app.post("/ingest_scan")
def ingest_scan():
    data = request.get_json(force=True, silent=True) or {}
    hub.ingest_scan(data)
    return jsonify({"ok": True, "n": int((data.get("n") or len(data.get("points", [])) or 0))})

@app.post("/api/pose")
def api_pose():
    data = request.get_json(force=True, silent=True) or {}
    x = float(data.get("x", 0.0))
    y = float(data.get("y", 0.0))
    yaw = float(data.get("yaw_deg", 0.0))
    hub.update_pose(x, y, yaw)
    return jsonify({"ok": True})

if __name__ == "__main__":
    # threaded=True so endpoints responsive while threads running
    app.run(host=HOST, port=PORT, threaded=True)
