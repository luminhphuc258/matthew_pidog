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


# =========================
# Helpers
# =========================

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


# =========================
# Config / Result
# =========================

@dataclass
class HandCfg:
    cam_dev: str = "/dev/video0"
    w: int = 640
    h: int = 480
    fps: int = 30

    # mediapipe
    det_conf: float = 0.55
    track_conf: float = 0.50
    max_hands: int = 1

    # performance
    process_every: int = 2

    # cooldown chống spam command (áp dụng cho fire action)
    action_cooldown_sec: float = 0.45

    # ===== Still-hand gating =====
    require_still: bool = True
    still_max_speed: float = 0.06

    # ===== STOP (fist tolerant) =====
    fist_curl_min: float = 0.74
    fist_need_extended_max: int = 1
    fist_single_finger_max_ratio: float = 1.20

    # ===== WAVE (OPEN/CLOSE sequence) =====
    wave_open_min_fingers: int = 3      # OPEN: 3..5
    wave_close_max_fingers: int = 1     # CLOSE: 0..1
    wave_window_sec: float = 3.5
    wave_need_toggles: int = 3
    wave_need_cycles: int = 2
    wave_lock_sec: float = 1.0

    # gating: khi đang "có vẻ wave" thì chặn phần khác
    wave_preempt: bool = True
    wave_preempt_min_events: int = 1

    # ===== TWO FINGERS (index+middle up) => stopmusic =====
    twof_require_index_middle: bool = True
    twof_allow_thumb: bool = True      # cho phép thumb thò ra vẫn tính
    twof_min_vertical_ratio: float = 1.25  # |dy| > ratio*|dx|
    twof_need_up: bool = True          # dy < 0

    # ===== Hand position regions (wrist x/y normalized 0..1) =====
    pos_left_x: float = 0.18
    pos_right_x: float = 0.82
    pos_up_y: float = 0.22
    pos_down_y: float = 0.78
    pos_hold_frames: int = 4

    # drawing
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


# =========================
# HandCommand
# =========================

class HandCommand:
    """
    Gestures (new):
      - STANDUP (hand high)
      - SIT (hand low)
      - MOVELEFT (hand too left)
      - MOVERIGHT (hand too right)
      - STOP (fist tolerant)
      - STOPMUSIC (two fingers up OR wave)
      - (keep: WAVE)

    Notes:
      - Keep draw_on_frame y chang.
      - WAVE + TWO_FINGERS => STOPMUSIC
      - Hand position uses WRIST (landmark 0) normalized.
    """

    # gestures
    G_NONE = "NONE"
    G_MOVELEFT = "MOVELEFT"
    G_MOVERIGHT = "MOVERIGHT"
    G_STANDUP = "STANDUP"
    G_SIT = "SIT"
    G_STOP = "STOP"
    G_WAVE = "WAVE"
    G_TWOFINGER = "TWOFINGER"
    G_STOPMUSIC = "STOPMUSIC"

    # wave states
    W_UNKNOWN = "UNK"
    W_OPEN = "OPEN"
    W_CLOSE = "CLOSE"
    W_MID = "MID"

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

        # runtime
        self._enabled = False
        self._running = False
        self._thr: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._last = HandLast(enabled=False, robot_state="UNKNOWN", ts=_now())
        self._last_action_ts = 0.0

        # last mediapipe landmarks
        self._last_mp_landmarks = None
        self._last_pts: Optional[List[Tuple[float, float, float]]] = None

        # fps
        self._fps_ts = _now()
        self._fps_n = 0
        self._fps_val = 0.0

        # wrist speed (still-hand)
        self._last_wrist_xy: Optional[Tuple[float, float]] = None

        # wave events
        self._wave_events: List[Tuple[float, str]] = []
        self._wave_last_state: str = self.W_UNKNOWN
        self._wave_lock_until: float = 0.0

        # position debounce
        self._pos_last: Optional[str] = None
        self._pos_hold: int = 0

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

    # -------------------------
    # Memory / state
    # -------------------------

    def _write_memory(self, gesture: str, action: str, face: str, bark: bool):
        obj = {
            "ts": _now(),
            "gesture": gesture,
            "action": action,
            "face": face,
            "bark": bool(bark),
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

        # reset states
        self._last_wrist_xy = None
        self._wave_events.clear()
        self._wave_last_state = self.W_UNKNOWN
        self._wave_lock_until = 0.0
        self._pos_last = None
        self._pos_hold = 0

        if on:
            self._update_last(gesture=None, action=None, face=None, bark=False, err=None)

    # -------------------------
    # Run loop
    # -------------------------

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

    # -------------------------
    # Drawing (KEEP SAME)
    # -------------------------

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
            cv2.rectangle(frame_bgr, (10, 10), (640, 62), (0, 0, 0), -1)
            cv2.putText(frame_bgr, txt, (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception:
            pass

        return frame_bgr

    # -------------------------
    # Internal loop
    # -------------------------

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

    # -------------------------
    # Geometry helpers
    # -------------------------

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

    def _vec_is_up(self, pts, mcp, tip) -> bool:
        mx, my, _ = pts[mcp]
        tx, ty, _ = pts[tip]
        dx = tx - mx
        dy = ty - my
        adx = abs(dx)
        ady = abs(dy)
        if ady <= self.cfg.twof_min_vertical_ratio * adx:
            return False
        if self.cfg.twof_need_up and dy >= 0:
            return False
        return True

    # -------------------------
    # Wave by OPEN/CLOSE sequence
    # -------------------------

    def _wave_state_from_fingers(self, finger_count: int) -> str:
        if finger_count >= self.cfg.wave_open_min_fingers:
            return self.W_OPEN
        if finger_count <= self.cfg.wave_close_max_fingers:
            return self.W_CLOSE
        return self.W_MID

    def _wave_push_event_if_needed(self, st: str):
        now = _now()
        cut = now - self.cfg.wave_window_sec
        while self._wave_events and self._wave_events[0][0] < cut:
            self._wave_events.pop(0)

        if st not in (self.W_OPEN, self.W_CLOSE):
            return
        if self._wave_last_state == st:
            return

        self._wave_last_state = st
        self._wave_events.append((now, st))

        while self._wave_events and self._wave_events[0][0] < cut:
            self._wave_events.pop(0)

    def _wave_detect(self) -> bool:
        now = _now()
        cut = now - self.cfg.wave_window_sec
        events = [(t, s) for (t, s) in self._wave_events if t >= cut]

        if len(events) < 2:
            return False

        toggles = 0
        for (_, a), (_, b) in zip(events[:-1], events[1:]):
            if a != b:
                toggles += 1
        if toggles < self.cfg.wave_need_toggles:
            return False

        states = [s for _, s in events]
        cycles = 0
        for i in range(len(states) - 2):
            if states[i] == self.W_OPEN and states[i + 1] == self.W_CLOSE and states[i + 2] == self.W_OPEN:
                cycles += 1
        return cycles >= self.cfg.wave_need_cycles

    # -------------------------
    # Gesture inference
    # -------------------------

    def _infer_gesture(self, frame_bgr) -> Tuple[str, Dict[str, Any]]:
        cv2 = self._cv2
        if cv2 is None:
            return self.G_NONE, {}

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)

        if not res.multi_hand_landmarks:
            self._last_mp_landmarks = None
            self._last_pts = None
            self._last_wrist_xy = None
            self._pos_last = None
            self._pos_hold = 0
            self._wave_events.clear()
            self._wave_last_state = self.W_UNKNOWN
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

        # extended by ratio
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

        # curl for fist
        c_idx = self._curl_score(pts, IN_TIP, IN_MCP, WRIST)
        c_mid = self._curl_score(pts, MD_TIP, MD_MCP, WRIST)
        c_rng = self._curl_score(pts, RG_TIP, RG_MCP, WRIST)
        c_pnk = self._curl_score(pts, PK_TIP, PK_MCP, WRIST)
        avg_curl = (c_idx + c_mid + c_rng + c_pnk) / 4.0

        info = {"finger_extended": finger_extended, "finger_count": finger_count}
        now = _now()

        # ===== wrist speed (still-hand) =====
        wx, wy, _ = pts[WRIST]
        speed = 0.0
        if self._last_wrist_xy is not None:
            px, py = self._last_wrist_xy
            speed = math.sqrt((wx - px) ** 2 + (wy - py) ** 2)
        self._last_wrist_xy = (wx, wy)

        if self.cfg.require_still and speed > self.cfg.still_max_speed:
            # đang rung tay mạnh -> bỏ qua tất cả (tránh spam)
            return self.G_NONE, info

        # ---- 1) WAVE lock ----
        if now < self._wave_lock_until:
            return self.G_STOPMUSIC, info

        # ---- 2) WAVE by OPEN/CLOSE sequence ----
        st = self._wave_state_from_fingers(finger_count)
        self._wave_push_event_if_needed(st)

        if self._wave_detect():
            self._wave_lock_until = now + self.cfg.wave_lock_sec
            self._pos_last = None
            self._pos_hold = 0
            return self.G_STOPMUSIC, info

        # wave preempt (chặn phần khác khi đang OPEN/CLOSE gần đây)
        if self.cfg.wave_preempt:
            cut = now - self.cfg.wave_window_sec
            recent_events = [1 for (t, _) in self._wave_events if t >= cut]
            if len(recent_events) >= self.cfg.wave_preempt_min_events and st in (self.W_OPEN, self.W_CLOSE):
                return self.G_NONE, info

        # ---- 3) TWO FINGERS (index+middle up) => STOPMUSIC ----
        # yêu cầu: 2 ngón tay đưa lên
        # - mặc định: index & middle extended
        # - ring & pinky phải cụp
        # - thumb có thể thò ra (tuỳ config)
        ok_two = False
        if self.cfg.twof_require_index_middle:
            if idx_ext and mid_ext and (not rng_ext) and (not pnk_ext):
                if (not th_ext) or self.cfg.twof_allow_thumb:
                    # thêm điều kiện vector "up"
                    if self._vec_is_up(pts, IN_MCP, IN_TIP) and self._vec_is_up(pts, MD_MCP, MD_TIP):
                        ok_two = True
        else:
            # fallback: finger_count == 2
            ok_two = (finger_count == 2)

        if ok_two:
            self._pos_last = None
            self._pos_hold = 0
            return self.G_STOPMUSIC, info

        # ---- 4) STOP (fist tolerant) ----
        # "nắm bàn tay": không thấy ngón, dư 1 ngón cũng ok (như code hiện tại)
        if avg_curl >= self.cfg.fist_curl_min and finger_count <= self.cfg.fist_need_extended_max:
            if finger_count == 1:
                ratios = []
                if finger_extended.get("index"):  ratios.append(r_idx)
                if finger_extended.get("middle"): ratios.append(r_mid)
                if finger_extended.get("ring"):   ratios.append(r_rng)
                if finger_extended.get("pinky"):  ratios.append(r_pnk)
                maxr = max(ratios) if ratios else 1.0
                if maxr <= self.cfg.fist_single_finger_max_ratio:
                    self._pos_last = None
                    self._pos_hold = 0
                    return self.G_STOP, info
            else:
                self._pos_last = None
                self._pos_hold = 0
                return self.G_STOP, info

        # ---- 5) Hand position regions (WRIST x/y) ----
        pos = None
        if wx <= self.cfg.pos_left_x:
            pos = self.G_MOVELEFT
        elif wx >= self.cfg.pos_right_x:
            pos = self.G_MOVERIGHT
        elif wy <= self.cfg.pos_up_y:
            pos = self.G_STANDUP
        elif wy >= self.cfg.pos_down_y:
            pos = self.G_SIT

        if pos is None:
            self._pos_last = None
            self._pos_hold = 0
            return self.G_NONE, info

        # debounce
        if pos == self._pos_last:
            self._pos_hold += 1
        else:
            self._pos_last = pos
            self._pos_hold = 1

        if self._pos_hold < self.cfg.pos_hold_frames:
            return self.G_NONE, info

        # confirmed
        self._pos_last = None
        self._pos_hold = 0
        return pos, info

    # -------------------------
    # Fire action (cooldown)
    # -------------------------

    def _maybe_fire_action(self, gesture: str):
        now = _now()
        if (now - self._last_action_ts) < self.cfg.action_cooldown_sec:
            return

        action, face, bark = self._map_gesture_to_action(gesture)
        if action is None:
            return

        try:
            self.on_action(action, face, bark)
        except Exception as e:
            self._update_last(err=f"on_action error: {e}", ts=_now())
            return

        self._last_action_ts = now
        self._write_memory(gesture, action, face, bark)
        self._update_last(action=action, face=face, bark=bark, ts=_now())
        print(f"[HandCommand] {gesture} => {action}", flush=True)

    def _map_gesture_to_action(self, gesture: str) -> Tuple[Optional[str], Optional[str], bool]:
        # action string để service publish MQTT
        if gesture == self.G_STOPMUSIC:
            return ("STOPMUSIC", "suprise", False)
        if gesture == self.G_STANDUP:
            return ("STANDUP", "suprise", False)
        if gesture == self.G_SIT:
            return ("SIT", "sleep", False)
        if gesture == self.G_MOVELEFT:
            return ("MOVELEFT", "what_is_it", False)
        if gesture == self.G_MOVERIGHT:
            return ("MOVERIGHT", "what_is_it", False)
        if gesture == self.G_STOP:
            return ("STOP", "sleep", False)

        return (None, None, False)
