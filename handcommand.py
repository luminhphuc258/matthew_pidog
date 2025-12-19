#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import time
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any, Dict, Tuple


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

    # detect
    det_conf: float = 0.55
    track_conf: float = 0.50
    max_hands: int = 1

    # performance: skip frames (1=every frame, 2=every 2 frames...)
    process_every: int = 2

    # cooldown: chống spam command
    action_cooldown_sec: float = 0.7

    # thresholds (tùy chỉnh dần)
    thumb_dir_deadzone: float = 0.10   # độ "nghiêng" tối thiểu để coi là thumbs left/right
    hand_high_y: float = 0.33          # palm y < 0.33 => hand high (stand)
    hand_low_y: float = 0.70           # palm y > 0.70 => hand low  (sit)

    # clap
    clap_window_sec: float = 0.35      # khoảng thời gian để coi 2 tay "vỗ"
    clap_dist_norm: float = 0.14       # khoảng cách 2 cổ tay (chuẩn hóa theo khung) để coi là "chạm"


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


# =========================
# HandCommand
# =========================

class HandCommand:
    """
    MediaPipe-based hand gesture recognizer.

    Public API:
      - start() / stop()
      - set_enabled(on: bool)
      - get_last() -> dict
      - draw_on_frame(frame_bgr) -> frame_bgr  (for WebDashboard)
    """

    # gesture names
    G_NONE = "NONE"
    G_BECKON = "BECKON"          # 4 fingers curling/waving
    G_INDEX_UP = "INDEX_UP"      # 1 index finger up
    G_PALM_HIGH = "PALM_HIGH"    # hand raised high -> stand
    G_PALM_LOW = "PALM_LOW"      # hand down low -> sit
    G_FIST = "FIST"              # stop
    G_CLAP = "CLAP"              # two hands clap -> back + bark
    G_THUMB_RIGHT = "THUMB_RIGHT"
    G_THUMB_LEFT = "THUMB_LEFT"

    # robot states saved in memory
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
        # optional boot helper for support_stand
        boot_helper: Optional[Any] = None,
        # optional camera supplier (if you already have frame pipeline)
        get_frame_bgr: Optional[Callable[[], Any]] = None,
        # optional: if you want open its own camera
        open_own_camera: bool = True,
        # memory file
        memory_file: str = "gesture_memory.json",
        # if True: remove old memory file on init
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

        self._last = HandLast(enabled=False, robot_state=self._read_robot_state(), ts=_now())
        self._last_action_ts = 0.0

        # last mediapipe landmarks (for draw_on_frame)
        self._last_mp_landmarks = None
        self._last_mp_handedness = None

        # fps calc
        self._fps_ts = _now()
        self._fps_n = 0
        self._fps_val = 0.0

        # mediapipe
        try:
            import mediapipe as mp
            self._mp = mp
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.cfg.max_hands,
                model_complexity=0,  # ✅ nhẹ nhất, chạy nhanh hơn
                min_detection_confidence=self.cfg.det_conf,
                min_tracking_confidence=self.cfg.track_conf,
            )
        except Exception as e:
            raise RuntimeError(
                "mediapipe is required for HandCommand.\n"
                "Install example: pip install mediapipe\n"
                f"Error: {e}"
            )

        # opencv only for drawing + camera if needed
        self._cv2 = None
        try:
            import cv2
            self._cv2 = cv2
        except Exception:
            self._cv2 = None

        # if open own camera
        self._cap = None

        # clap state (if you later enable multi-hand)
        self._last_clap_ts = 0.0

    # -------------------------
    # State / memory
    # -------------------------

    def _read_robot_state(self) -> str:
        data = _safe_read_json(self.mem_path)
        st = (data.get("robot_state") or "").upper().strip()
        if st:
            return st
        return self.S_UNKNOWN

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
            }

    def set_enabled(self, on: bool):
        on = bool(on)
        with self._lock:
            self._enabled = on
            self._last.enabled = on
            self._last.ts = _now()
        # nếu bật lên: clear last gesture cho sạch
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
    # Drawing (WebDashboard)
    # -------------------------

    def draw_on_frame(self, frame_bgr):
        """
        Draw mediapipe hand landmarks + gesture text onto BGR frame for WebDashboard.
        """
        if self._cv2 is None:
            return frame_bgr

        cv2 = self._cv2
        try:
            # draw landmarks
            lm = self._last_mp_landmarks
            if lm is not None:
                self._mp_draw.draw_landmarks(
                    frame_bgr,
                    lm,
                    self._mp_hands.HAND_CONNECTIONS
                )

            last = self.get_last()
            enabled = last.get("enabled", False)
            gest = last.get("gesture") or "NA"
            fps = last.get("fps", None)

            txt = ("HAND: ON  " if enabled else "HAND: OFF ") + f" {gest}"
            if isinstance(fps, (int, float)):
                txt += f"  ({fps:.1f}fps)"

            # background box
            cv2.rectangle(frame_bgr, (10, 10), (520, 60), (0, 0, 0), -1)
            cv2.putText(frame_bgr, txt, (20, 45),
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
                # update fps counter even if skipping (for UI feel)
                self._tick_fps()
                time.sleep(0.001)
                continue

            try:
                gesture = self._infer_gesture(frame)
                self._tick_fps()

                enabled = False
                with self._lock:
                    enabled = bool(self._enabled)

                if enabled and gesture and gesture != self.G_NONE:
                    self._maybe_fire_action(gesture)

                self._update_last(
                    enabled=enabled,
                    gesture=(gesture if gesture != self.G_NONE else None),
                    fps=self._fps_val,
                    ts=_now(),
                    err=None
                )
            except Exception as e:
                self._update_last(err=f"infer_error: {e}", ts=_now())

            dt = _now() - t0
            # giữ loop nhẹ nhàng
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
    # Gesture inference
    # -------------------------

    def _infer_gesture(self, frame_bgr) -> str:
        """
        Return one of gesture constants.
        """
        cv2 = self._cv2
        if cv2 is None:
            return self.G_NONE

        # convert BGR -> RGB for mediapipe
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            self._last_mp_landmarks = None
            self._last_mp_handedness = None
            return self.G_NONE

        lm = res.multi_hand_landmarks[0]
        self._last_mp_landmarks = lm
        try:
            self._last_mp_handedness = res.multi_handedness[0]
        except Exception:
            self._last_mp_handedness = None

        # normalized points
        pts = [(p.x, p.y, p.z) for p in lm.landmark]

        # key indexes (mediapipe)
        WRIST = 0
        TH_CMC = 1
        TH_MCP = 2
        TH_IP  = 3
        TH_TIP = 4
        IN_MCP = 5
        IN_PIP = 6
        IN_DIP = 7
        IN_TIP = 8
        MD_MCP = 9
        MD_PIP = 10
        MD_DIP = 11
        MD_TIP = 12
        RG_MCP = 13
        RG_PIP = 14
        RG_DIP = 15
        RG_TIP = 16
        PK_MCP = 17
        PK_PIP = 18
        PK_DIP = 19
        PK_TIP = 20

        def dist2(i, j) -> float:
            ax, ay, _ = pts[i]
            bx, by, _ = pts[j]
            dx = ax - bx
            dy = ay - by
            return dx*dx + dy*dy

        def is_finger_extended(tip, pip, mcp) -> bool:
            # y nhỏ hơn => cao hơn trong ảnh (tay hướng lên)
            ty = pts[tip][1]
            py = pts[pip][1]
            my = pts[mcp][1]
            return (ty < py) and (py < my)

        def finger_curl_score(tip, pip, mcp, wrist=WRIST) -> float:
            # lớn => đang co về cổ tay
            # dùng khoảng cách tip->wrist so với mcp->wrist
            tw = math.sqrt(dist2(tip, wrist))
            mw = math.sqrt(dist2(mcp, wrist))
            if mw <= 1e-6:
                return 0.0
            return _clamp((mw - tw) / mw, 0.0, 1.0)

        # palm center
        palm_x = (pts[WRIST][0] + pts[IN_MCP][0] + pts[MD_MCP][0] + pts[RG_MCP][0] + pts[PK_MCP][0]) / 5.0
        palm_y = (pts[WRIST][1] + pts[IN_MCP][1] + pts[MD_MCP][1] + pts[RG_MCP][1] + pts[PK_MCP][1]) / 5.0

        # --------- Palm high / low (priority: stand/sit) ---------
        if palm_y < self.cfg.hand_high_y:
            return self.G_PALM_HIGH
        if palm_y > self.cfg.hand_low_y:
            return self.G_PALM_LOW

        # --------- Finger extended flags ---------
        idx_ext = is_finger_extended(IN_TIP, IN_PIP, IN_MCP)
        mid_ext = is_finger_extended(MD_TIP, MD_PIP, MD_MCP)
        rng_ext = is_finger_extended(RG_TIP, RG_PIP, RG_MCP)
        pnk_ext = is_finger_extended(PK_TIP, PK_PIP, PK_MCP)

        ext_count = sum([idx_ext, mid_ext, rng_ext, pnk_ext])

        # --------- Fist ---------
        # nếu 4 ngón đều co mạnh -> fist
        c_idx = finger_curl_score(IN_TIP, IN_PIP, IN_MCP)
        c_mid = finger_curl_score(MD_TIP, MD_PIP, MD_MCP)
        c_rng = finger_curl_score(RG_TIP, RG_PIP, RG_MCP)
        c_pnk = finger_curl_score(PK_TIP, PK_PIP, PK_MCP)
        avg_curl = (c_idx + c_mid + c_rng + c_pnk) / 4.0

        if avg_curl > 0.68 and ext_count == 0:
            return self.G_FIST

        # --------- Index up ---------
        # chỉ index duỗi, các ngón khác co
        if idx_ext and (not mid_ext) and (not rng_ext) and (not pnk_ext):
            # thêm điều kiện các ngón kia co vừa đủ
            if (c_mid + c_rng + c_pnk) / 3.0 > 0.35:
                return self.G_INDEX_UP

        # --------- Beckon (4-finger curling) ---------
        # index+mid+ring+pinky curl khá cao, thumb không quan trọng
        if avg_curl > 0.45 and ext_count <= 1:
            # muốn beckon thì curl nhưng không phải fist quá chặt
            if 0.45 <= avg_curl <= 0.75:
                return self.G_BECKON

        # --------- Thumb left/right ---------
        # đo hướng vector thumb_mcp -> thumb_tip theo trục x
        th_dx = pts[TH_TIP][0] - pts[TH_MCP][0]
        th_dy = pts[TH_TIP][1] - pts[TH_MCP][1]
        # thumbs "ngang" hơn "dọc"
        if abs(th_dx) > abs(th_dy) and abs(th_dx) > self.cfg.thumb_dir_deadzone:
            if th_dx > 0:
                return self.G_THUMB_RIGHT
            else:
                return self.G_THUMB_LEFT

        return self.G_NONE

    # -------------------------
    # Action mapping + safety state
    # -------------------------

    def _maybe_fire_action(self, gesture: str):
        now = _now()
        if (now - self._last_action_ts) < self.cfg.action_cooldown_sec:
            return

        # read last robot state
        robot_state = self._read_robot_state()

        action, face, bark, next_state = self._map_gesture_to_action(gesture, robot_state)
        if action is None:
            return

        # safety: nếu đang SIT mà cần di chuyển/stand -> support stand trước
        # (bạn có thể mở rộng thêm state khác)
        needs_standing_first = action in ("FORWARD", "TROT_FORWARD", "BACK", "TURN_LEFT", "TURN_RIGHT", "STAND")
        if needs_standing_first and robot_state == self.S_SIT:
            if self.boot_helper is not None and hasattr(self.boot_helper, "support_stand"):
                try:
                    print("[HandCommand] safety: robot SIT -> support_stand() before action")
                    self.boot_helper.support_stand()
                    robot_state = self.S_STAND
                except Exception as e:
                    print("[HandCommand] support_stand error:", e, flush=True)

        # fire action
        try:
            self.on_action(action, face, bark)
        except Exception as e:
            self._update_last(err=f"on_action error: {e}", ts=_now())
            return

        self._last_action_ts = now

        # update memory + last
        self._write_memory(gesture, action, face, bark, next_state)
        self._update_last(action=action, face=face, bark=bark, robot_state=next_state, ts=_now())

        print(f"[HandCommand] GESTURE={gesture} => ACTION={action} FACE={face} bark={bark}", flush=True)

    def _map_gesture_to_action(self, gesture: str, robot_state: str) -> Tuple[Optional[str], Optional[str], bool, str]:
        """
        Returns: (action, face, bark, next_robot_state)
        """
        # mapping theo yêu cầu bạn
        if gesture == self.G_BECKON:
            return ("FORWARD", "suprise", False, self.S_MOVING)

        if gesture == self.G_INDEX_UP:
            return ("TROT_FORWARD", "suprise", False, self.S_MOVING)

        if gesture == self.G_PALM_HIGH:
            # đứng lên
            return ("STAND", "what_is_it", False, self.S_STAND)

        if gesture == self.G_PALM_LOW:
            return ("SIT", "sad", False, self.S_SIT)

        if gesture == self.G_FIST:
            return ("STOP", "sleep", False, self.S_STOP)

        if gesture == self.G_THUMB_RIGHT:
            return ("TURN_RIGHT", "what_is_it", False, self.S_MOVING)

        if gesture == self.G_THUMB_LEFT:
            return ("TURN_LEFT", "suprise", False, self.S_MOVING)

        # Clap: hiện tại file này chạy max_hands=1 cho nhanh,
        # nên CLAP chưa kích hoạt trong _infer_gesture().
        # Nếu bạn muốn clap thật (2 tay), mình sẽ nâng cấp phiên bản max_hands=2 + wrist distance.
        if gesture == self.G_CLAP:
            return ("BACK", "angry", True, self.S_MOVING)

        return (None, None, False, robot_state)
