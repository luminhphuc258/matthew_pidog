#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any, Dict, Tuple, List


def _now() -> float:
    return time.time()


def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8", errors="ignore") or "{}")
    except Exception:
        return {}


def _safe_write_json(path: Path, obj: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _rm_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


@dataclass
class HandCfg:
    cam_dev: str = "/dev/video0"
    w: int = 640
    h: int = 480
    fps: int = 30

    det_conf: float = 0.55
    track_conf: float = 0.50
    max_hands: int = 1
    process_every: int = 2

    action_cooldown_sec: float = 0.45

    # ===== Direction by finger vectors (MCP->TIP) =====
    dir_min_len: float = 0.10      # ignore short vectors (curled fingers)
    dir_axis_ratio: float = 1.25   # |dy| > r*|dx| => vertical, |dx| > r*|dy| => horizontal
    dir_votes_need: int = 1        # minimum votes to accept direction
    prefer_vertical: bool = True   # UP/DOWN preferred over LEFT/RIGHT

    # ===== Fist / STOP =====
    fist_curl_min: float = 0.74
    fist_need_extended_max: int = 0  # 0: no finger extended

    # ===== WAVE (6th class) =====
    # only when ALL 5 fingers open
    wave_window_sec: float = 0.90
    wave_min_amplitude_y: float = 0.08
    wave_min_amplitude_x: float = 0.10
    wave_min_swings: int = 2
    wave_lock_sec: float = 0.80      # lock wave for stable UI

    # ===== Drawing =====
    draw_tip_radius: int = 7
    draw_line_thick: int = 3


@dataclass
class HandLast:
    enabled: bool = False
    gesture: Optional[str] = None
    action: Optional[str] = None
    face: Optional[str] = None
    bark: bool = False
    fps: float = 0.0
    robot_state: str = "UNKNOWN"
    ts: float = 0.0
    err: Optional[str] = None

    finger_extended: Dict[str, bool] = field(default_factory=dict)
    finger_count: int = 0


class HandCommand:
    """
    6-class gestures:
      - LEFT, RIGHT, UP, DOWN, STOP, WAVE
    """

    G_NONE = "NONE"
    G_LEFT = "LEFT"
    G_RIGHT = "RIGHT"
    G_UP = "UP"
    G_DOWN = "DOWN"
    G_STOP = "STOP"
    G_WAVE = "WAVE"

    S_STAND = "STAND"
    S_SIT = "SIT"
    S_LYING = "LYING"
    S_MOVING = "MOVING"
    S_STOP = "STOP"
    S_UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        cfg: HandCfg,
        on_action: Callable[[str, str, bool], None],
        boot_helper: Optional[Any] = None,
        get_frame_bgr: Optional[Callable[[], Any]] = None,
        open_own_camera: bool = True,
        memory_file: str = "gesture_memory.json",
        clear_memory_on_start: bool = True,
    ):
        self.cfg = cfg
        self.on_action = on_action
        self.boot_helper = boot_helper
        self.get_frame_bgr = get_frame_bgr
        self.open_own_camera = bool(open_own_camera)

        self.base_dir = Path(__file__).resolve().parent
        self.mem_path = self.base_dir / memory_file
        if clear_memory_on_start:
            _rm_if_exists(self.mem_path)

        self._enabled = False
        self._running = False
        self._thr: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._last = HandLast(enabled=False, robot_state=self._read_robot_state(), ts=_now())
        self._last_action_ts = 0.0

        # last mediapipe
        self._last_mp_landmarks = None
        self._last_pts: Optional[List[Tuple[float, float, float]]] = None

        # fps
        self._fps_ts = _now()
        self._fps_n = 0
        self._fps_val = 0.0

        # wave history + lock
        self._hist_x: List[Tuple[float, float]] = []
        self._hist_y: List[Tuple[float, float]] = []
        self._wave_lock_until: float = 0.0

        # mediapipe
        try:
            import mediapipe as mp
            self._mp = mp
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.cfg.max_hands,
                model_complexity=0,
                min_detection_confidence=self.cfg.det_conf,
                min_tracking_confidence=self.cfg.track_conf,
            )
        except Exception as e:
            raise RuntimeError(
                "mediapipe is required for HandCommand.\n"
                "Install: pip install mediapipe\n"
                f"Error: {e}"
            )

        # opencv
        self._cv2 = None
        try:
            import cv2
            self._cv2 = cv2
        except Exception:
            self._cv2 = None

        self._cap = None

    # ----------------- memory/state -----------------

    def _read_robot_state(self) -> str:
        data = _safe_read_json(self.mem_path)
        st = (data.get("robot_state") or "").upper().strip()
        return st or self.S_UNKNOWN

    def _write_memory(self, gesture: str, action: str, face: str, bark: bool, robot_state: str):
        obj = {
            "ts": _now(),
            "gesture": gesture,
            "action": action,
            "face": face,
            "bark": bool(bark),
            "robot_state": robot_state,
        }
        _safe_write_json(self.mem_path, obj)

    def _update_last(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self._last, k, v)

    def get_last(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._last.enabled,
                "gesture": self._last.gesture,
                "action": self._last.action,
                "face": self._last.face,
                "bark": self._last.bark,
                "fps": self._last.fps,
                "robot_state": self._last.robot_state,
                "ts": self._last.ts,
                "err": self._last.err,
                "finger_extended": dict(self._last.finger_extended),
                "finger_count": int(self._last.finger_count),
            }

    def set_enabled(self, on: bool):
        on = bool(on)
        with self._lock:
            self._enabled = on
            self._last.enabled = on
            self._last.ts = _now()

        self._hist_x.clear()
        self._hist_y.clear()
        self._wave_lock_until = 0.0

        if on:
            self._update_last(gesture=None, action=None, face=None, bark=False, err=None)

    # ----------------- run loop -----------------

    def start(self):
        if self._running:
            return
        self._running = True

        if self.open_own_camera and self.get_frame_bgr is None:
            if self._cv2 is None:
                self._running = False
                raise RuntimeError("opencv-python is required to open camera in HandCommand")
            self._cap = self._cv2.VideoCapture(self.cfg.cam_dev)
            self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.cfg.w)
            self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.h)
            self._cap.set(self._cv2.CAP_PROP_FPS, self.cfg.fps)

        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._running = False
        if self._thr:
            self._thr.join(timeout=1.0)
        self._thr = None
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    # ----------------- drawing -----------------

    def draw_on_frame(self, frame_bgr):
        if self._cv2 is None:
            return frame_bgr
        cv2 = self._cv2
        try:
            last = self.get_last()
            enabled = last.get("enabled", False)
            gest = last.get("gesture") or "NA"
            fps = last.get("fps", None)
            fcount = last.get("finger_count", 0)
            fext = last.get("finger_extended", {}) or {}

            lm = self._last_mp_landmarks
            if lm is not None:
                self._mp_draw.draw_landmarks(frame_bgr, lm, self._mp_hands.HAND_CONNECTIONS)

            pts = self._last_pts
            if pts is not None:
                H, W = frame_bgr.shape[:2]

                def P(i):
                    x, y, _ = pts[i]
                    return int(x * W), int(y * H)

                TH_MCP, TH_IP, TH_TIP = 2, 3, 4
                IN_MCP, IN_PIP, IN_DIP, IN_TIP = 5, 6, 7, 8
                MD_MCP, MD_PIP, MD_DIP, MD_TIP = 9, 10, 11, 12
                RG_MCP, RG_PIP, RG_DIP, RG_TIP = 13, 14, 15, 16
                PK_MCP, PK_PIP, PK_DIP, PK_TIP = 17, 18, 19, 20

                fingers = {
                    "thumb": [TH_MCP, TH_IP, TH_TIP],
                    "index": [IN_MCP, IN_PIP, IN_DIP, IN_TIP],
                    "middle": [MD_MCP, MD_PIP, MD_DIP, MD_TIP],
                    "ring": [RG_MCP, RG_PIP, RG_DIP, RG_TIP],
                    "pinky": [PK_MCP, PK_PIP, PK_DIP, PK_TIP],
                }

                for name, chain in fingers.items():
                    on = bool(fext.get(name, False))
                    col = (0, 255, 0) if on else (80, 80, 80)
                    for a, b in zip(chain[:-1], chain[1:]):
                        cv2.line(frame_bgr, P(a), P(b), col, self.cfg.draw_line_thick)
                    cv2.circle(frame_bgr, P(chain[-1]), self.cfg.draw_tip_radius, col, -1)

            txt = ("HAND: ON  " if enabled else "HAND: OFF ") + f"{gest}  fingers:{fcount}"
            if isinstance(fps, (int, float)):
                txt += f"  ({fps:.1f}fps)"
            cv2.rectangle(frame_bgr, (10, 10), (620, 62), (0, 0, 0), -1)
            cv2.putText(frame_bgr, txt, (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception:
            pass
        return frame_bgr

    # ----------------- internals -----------------

    def _grab_frame(self):
        if self.get_frame_bgr is not None:
            try:
                return self.get_frame_bgr()
            except Exception:
                return None
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def _loop(self):
        frame_i = 0
        while self._running:
            t0 = _now()
            frame = self._grab_frame()
            if frame is None:
                self._update_last(err="no_frame", ts=_now())
                time.sleep(0.03)
                continue

            frame_i += 1
            if self.cfg.process_every > 1 and (frame_i % self.cfg.process_every != 0):
                self._tick_fps()
                time.sleep(0.001)
                continue

            try:
                gesture, info = self._infer_gesture(frame)
                self._tick_fps()

                with self._lock:
                    enabled = bool(self._enabled)

                if enabled and gesture and gesture != self.G_NONE:
                    self._maybe_fire_action(gesture)

                self._update_last(
                    enabled=enabled,
                    gesture=(gesture if gesture != self.G_NONE else None),
                    fps=self._fps_val,
                    ts=_now(),
                    err=None,
                    finger_extended=info.get("finger_extended", {}),
                    finger_count=info.get("finger_count", 0),
                )
            except Exception as e:
                self._update_last(err=f"infer_error: {e}", ts=_now())

            dt = _now() - t0
            if dt < 0.01:
                time.sleep(0.002)

    def _tick_fps(self):
        self._fps_n += 1
        dt = _now() - self._fps_ts
        if dt >= 1.0:
            self._fps_val = float(self._fps_n) / dt
            self._fps_n = 0
            self._fps_ts = _now()

    # ---- geometry ----

    def _dist2_xy(self, pts, i, j) -> float:
        ax, ay, _ = pts[i]
        bx, by, _ = pts[j]
        dx = ax - bx
        dy = ay - by
        return dx * dx + dy * dy

    def _finger_ratio(self, pts, tip, mcp, wrist=0) -> float:
        tw = math.sqrt(self._dist2_xy(pts, tip, wrist))
        mw = math.sqrt(self._dist2_xy(pts, mcp, wrist))
        if mw <= 1e-6:
            return 1.0
        return tw / mw

    def _curl_score(self, pts, tip, mcp, wrist=0) -> float:
        r = self._finger_ratio(pts, tip, mcp, wrist)
        return _clamp((1.05 - r) / 0.35, 0.0, 1.0)

    def _thumb_extended(self, pts) -> bool:
        WRIST = 0
        TH_TIP = 4
        IN_MCP = 5
        d_tw = math.sqrt(self._dist2_xy(pts, TH_TIP, WRIST))
        d_ti = math.sqrt(self._dist2_xy(pts, TH_TIP, IN_MCP))
        return (d_tw > 0.16) and (d_ti > 0.09)

    def _wave_check(self, hist: List[Tuple[float, float]], v: float, amp_thresh: float) -> bool:
        now = _now()
        hist.append((now, v))
        cut = now - self.cfg.wave_window_sec
        while hist and hist[0][0] < cut:
            hist.pop(0)

        if len(hist) < 6:
            return False

        vs = [x for _, x in hist]
        amp = max(vs) - min(vs)
        if amp < amp_thresh:
            return False

        m = sum(vs) / len(vs)
        signs = []
        for x in vs:
            d = x - m
            if abs(d) < 0.012:
                continue
            signs.append(1 if d > 0 else -1)

        if len(signs) < 4:
            return False

        changes = 0
        for a, b in zip(signs[:-1], signs[1:]):
            if a != b:
                changes += 1

        return changes >= self.cfg.wave_min_swings

    def _finger_dir_vote(self, pts) -> Optional[str]:
        IN_MCP, IN_TIP = 5, 8
        MD_MCP, MD_TIP = 9, 12
        RG_MCP, RG_TIP = 13, 16
        PK_MCP, PK_TIP = 17, 20

        votes = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}

        def vote(mcp, tip):
            mx, my, _ = pts[mcp]
            tx, ty, _ = pts[tip]
            dx = tx - mx
            dy = ty - my

            ln = math.sqrt(dx * dx + dy * dy)
            if ln < self.cfg.dir_min_len:
                return

            adx = abs(dx)
            ady = abs(dy)

            if ady > self.cfg.dir_axis_ratio * adx:
                if dy < 0:
                    votes["UP"] += 1
                else:
                    votes["DOWN"] += 1
            elif adx > self.cfg.dir_axis_ratio * ady:
                if dx > 0:
                    votes["RIGHT"] += 1
                else:
                    votes["LEFT"] += 1

        vote(IN_MCP, IN_TIP)
        vote(MD_MCP, MD_TIP)
        vote(RG_MCP, RG_TIP)
        vote(PK_MCP, PK_TIP)

        # accept only if votes enough
        best_dir, best_n = max(votes.items(), key=lambda kv: kv[1])
        if best_n < self.cfg.dir_votes_need:
            return None

        # resolve by preference
        updown = max(votes["UP"], votes["DOWN"])
        leftright = max(votes["LEFT"], votes["RIGHT"])

        if self.cfg.prefer_vertical and updown >= leftright:
            if votes["UP"] >= votes["DOWN"] and votes["UP"] >= self.cfg.dir_votes_need:
                return "UP"
            if votes["DOWN"] >= self.cfg.dir_votes_need:
                return "DOWN"
        else:
            if votes["LEFT"] >= votes["RIGHT"] and votes["LEFT"] >= self.cfg.dir_votes_need:
                return "LEFT"
            if votes["RIGHT"] >= self.cfg.dir_votes_need:
                return "RIGHT"

        # fallback
        return best_dir

    # ----------------- inference -----------------

    def _infer_gesture(self, frame_bgr) -> Tuple[str, Dict[str, Any]]:
        cv2 = self._cv2
        if cv2 is None:
            return self.G_NONE, {}

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)

        if not res.multi_hand_landmarks:
            self._last_mp_landmarks = None
            self._last_pts = None
            self._hist_x.clear()
            self._hist_y.clear()
            return self.G_NONE, {}

        lm = res.multi_hand_landmarks[0]
        self._last_mp_landmarks = lm
        pts = [(p.x, p.y, p.z) for p in lm.landmark]
        self._last_pts = pts

        WRIST = 0
        IN_MCP, IN_TIP = 5, 8
        MD_MCP, MD_TIP = 9, 12
        RG_MCP, RG_TIP = 13, 16
        PK_MCP, PK_TIP = 17, 20

        # extended detection
        ext_thr = 1.08
        r_idx = self._finger_ratio(pts, IN_TIP, IN_MCP, WRIST)
        r_mid = self._finger_ratio(pts, MD_TIP, MD_MCP, WRIST)
        r_rng = self._finger_ratio(pts, RG_TIP, RG_MCP, WRIST)
        r_pnk = self._finger_ratio(pts, PK_TIP, PK_MCP, WRIST)

        idx_ext = (r_idx >= ext_thr)
        mid_ext = (r_mid >= ext_thr)
        rng_ext = (r_rng >= ext_thr)
        pnk_ext = (r_pnk >= ext_thr)
        th_ext = self._thumb_extended(pts)

        finger_extended = {
            "thumb": th_ext,
            "index": idx_ext,
            "middle": mid_ext,
            "ring": rng_ext,
            "pinky": pnk_ext,
        }
        finger_count = sum(1 for v in finger_extended.values() if v)

        # curl for STOP (fist)
        c_idx = self._curl_score(pts, IN_TIP, IN_MCP, WRIST)
        c_mid = self._curl_score(pts, MD_TIP, MD_MCP, WRIST)
        c_rng = self._curl_score(pts, RG_TIP, RG_MCP, WRIST)
        c_pnk = self._curl_score(pts, PK_TIP, PK_MCP, WRIST)
        avg_curl = (c_idx + c_mid + c_rng + c_pnk) / 4.0

        info = {"finger_extended": finger_extended, "finger_count": finger_count}
        now = _now()

        # ---- 1) WAVE lock ----
        if now < self._wave_lock_until:
            return self.G_WAVE, info

        # ---- 2) WAVE detect only when 5 fingers open ----
        if finger_count >= 5:
            wx, wy, _ = pts[WRIST]
            # use wrist as stable anchor
            if self._wave_check(self._hist_y, wy, self.cfg.wave_min_amplitude_y) or \
               self._wave_check(self._hist_x, wx, self.cfg.wave_min_amplitude_x):
                self._wave_lock_until = now + self.cfg.wave_lock_sec
                return self.G_WAVE, info
        else:
            self._hist_x.clear()
            self._hist_y.clear()

        # ---- 3) STOP (fist) ----
        if avg_curl >= self.cfg.fist_curl_min and finger_count <= self.cfg.fist_need_extended_max:
            return self.G_STOP, info

        # ---- 4) Direction by finger vectors ----
        d = self._finger_dir_vote(pts)
        if d == "UP":
            return self.G_UP, info
        if d == "DOWN":
            return self.G_DOWN, info
        if d == "LEFT":
            return self.G_LEFT, info
        if d == "RIGHT":
            return self.G_RIGHT, info

        return self.G_NONE, info

    # ----------------- action mapping -----------------

    def _maybe_fire_action(self, gesture: str):
        now = _now()
        if (now - self._last_action_ts) < self.cfg.action_cooldown_sec:
            return

        robot_state = self._read_robot_state()
        action, face, bark, next_state = self._map_gesture_to_action(gesture, robot_state)
        if action is None:
            return

        # safety if sitting
        needs_standing_first = action in ("FORWARD", "BACK", "TURN_LEFT", "TURN_RIGHT", "TROT_FORWARD")
        if needs_standing_first and robot_state == self.S_SIT:
            if self.boot_helper is not None and hasattr(self.boot_helper, "support_stand"):
                try:
                    self.boot_helper.support_stand()
                    robot_state = self.S_STAND
                except Exception:
                    pass

        try:
            self.on_action(action, face, bark)
        except Exception as e:
            self._update_last(err=f"on_action error: {e}", ts=_now())
            return

        self._last_action_ts = now
        self._write_memory(gesture, action, face, bark, next_state)
        self._update_last(action=action, face=face, bark=bark, robot_state=next_state, ts=_now())
        print(f"[HandCommand] {gesture} => {action}", flush=True)

    def _map_gesture_to_action(self, gesture: str, robot_state: str) -> Tuple[Optional[str], Optional[str], bool, str]:
        # mapping bạn có thể đổi tuỳ ý
        if gesture == self.G_UP:
            return ("FORWARD", "suprise", False, self.S_MOVING)
        if gesture == self.G_DOWN:
            return ("BACK", "angry", False, self.S_MOVING)
        if gesture == self.G_LEFT:
            return ("TURN_LEFT", "suprise", False, self.S_MOVING)
        if gesture == self.G_RIGHT:
            return ("TURN_RIGHT", "what_is_it", False, self.S_MOVING)
        if gesture == self.G_STOP:
            return ("STOP", "sleep", False, self.S_STOP)
        if gesture == self.G_WAVE:
            return ("TROT_FORWARD", "suprise", False, self.S_MOVING)

        return (None, None, False, robot_state)
