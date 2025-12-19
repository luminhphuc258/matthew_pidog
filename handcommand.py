# handcommand.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import math
import threading
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# Optional: mediapipe
try:
    import mediapipe as mp
except Exception:
    mp = None


# =========================
# Utils
# =========================
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _now():
    return time.time()


def _dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _safe_json_load(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
    return default


def _safe_json_dump(path: Path, obj):
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _append_jsonl(path: Path, obj):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


# =========================
# Config
# =========================
@dataclass
class HandCmdCfg:
    cam_dev: str = "/dev/video0"
    cam_w: int = 640
    cam_h: int = 480
    cam_fps: int = 30

    # detection
    min_det_conf: float = 0.6
    min_track_conf: float = 0.6

    # avoid spam
    cooldown_sec: float = 0.9
    same_gesture_hold_sec: float = 0.25
    max_fps: float = 20.0

    # file memory
    state_file: str = "gesture_state.json"
    log_file: str = "gesture_log.jsonl"

    # ✅ NEW: clear old memory on start
    clear_old_memory_on_start: bool = True

    # face3d UDP
    face_host: str = "127.0.0.1"
    face_port: int = 39393

    # debug overlay
    draw_overlay: bool = True


# =========================
# HandCommand
# =========================
class HandCommand:
    """
    HandCommand: nhận diện gesture realtime từ live camera.

    Output: gọi callback on_cmd(move_str) + set_face(emo_str)

    - Có state machine cơ bản + ghi file gesture_state.json và gesture_log.jsonl
    - Có check robot_state trước khi chạy lệnh để tránh té:
        * nếu robot đang SIT mà cần MOVE/STAND => gọi support_stand() trước
    - Có thể bật/tắt bằng set_enabled(True/False)

    Gesture (heuristic, mediapipe):
      1) Beckon 4 fingers wave (come closer) => FORWARD + face suprise
      2) Index up to camera => TROT_FORWARD + face suprise
      3) Palm high => STAND + face suprise
      4) Palm low => SIT + face sad
      5) Fist => STOP + face sleep
      6) Clap (2 hands) => BACK + BARK + face angry
      7) Thumb right => TURN_RIGHT + face what_is_it
      8) Thumb left => TURN_LEFT + face suprise

    NOTE:
      - Bạn map các string command này ở main bằng MotionController của bạn.
    """

    VALID_STATES = {"UNKNOWN", "SIT", "STAND", "MOVING", "STOPPED"}

    def __init__(
        self,
        cfg: HandCmdCfg,
        on_cmd: Optional[Callable[[str], None]] = None,
        set_face: Optional[Callable[[str], None]] = None,

        # optional: bark callback (hoặc bạn dùng ActiveListenerV2 tự bark)
        on_bark: Optional[Callable[[], None]] = None,

        # optional: support stand helper (MatthewPidogBootClass instance)
        boot_helper: Optional[Any] = None,
    ):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for HandCommand.")

        if mp is None:
            raise RuntimeError(
                "mediapipe is required for HandCommand.\n"
                "Install: pip install mediapipe"
            )

        self.cfg = cfg
        self.on_cmd = on_cmd
        self.set_face_cb = set_face
        self.on_bark = on_bark
        self.boot_helper = boot_helper

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._enabled_lock = threading.Lock()
        self._enabled = False

        # face udp
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_addr = (self.cfg.face_host, int(self.cfg.face_port))

        # files
        self._state_path = Path(self.cfg.state_file)
        self._log_path = Path(self.cfg.log_file)

        # ✅ NEW: clear old memory on start
        if getattr(self.cfg, "clear_old_memory_on_start", True):
            try:
                if self._state_path.exists():
                    self._state_path.unlink()
            except Exception:
                pass
            try:
                if self._log_path.exists():
                    self._log_path.unlink()
            except Exception:
                pass

        # state
        self._state = self._load_state()
        if self._state.get("robot_state") not in self.VALID_STATES:
            self._state["robot_state"] = "UNKNOWN"

        # recent gesture buffer
        self._last_emit_ts = 0.0
        self._last_gesture = ""
        self._gesture_since = 0.0

        # clap detection memory
        self._last_clap_close_ts = 0.0

        # mediapipe hands
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=float(self.cfg.min_det_conf),
            min_tracking_confidence=float(self.cfg.min_track_conf),
        )

        self._cap = None
        self._last_frame = None
        self._last_frame_lock = threading.Lock()

    # ---------------- public ----------------
    def start(self, enabled: bool = True):
        self.set_enabled(enabled)
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="HandCommand", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._thread:
                self._thread.join(timeout=1.5)
        except Exception:
            pass
        self._release_cam()

    def join(self, timeout: float = 2.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    def set_enabled(self, on: bool):
        with self._enabled_lock:
            self._enabled = bool(on)

    def is_enabled(self) -> bool:
        with self._enabled_lock:
            return bool(self._enabled)

    def get_last_frame(self):
        with self._last_frame_lock:
            return None if self._last_frame is None else self._last_frame.copy()

    # ---------------- face ----------------
    def send_face_cmd(self, msg: str):
        try:
            self._udp.sendto(msg.encode("utf-8"), self._udp_addr)
        except Exception:
            pass

    def set_face(self, emo: str):
        if self.set_face_cb:
            try:
                self.set_face_cb(emo)
                return
            except Exception:
                pass
        # fallback to udp
        self.send_face_cmd(f"EMO {emo}")

    # ---------------- file memory ----------------
    def _load_state(self) -> Dict[str, Any]:
        default = {
            "robot_state": "UNKNOWN",
            "last_action": "",
            "last_face": "",
            "last_gesture": "",
            "ts": 0.0,
        }
        return _safe_json_load(self._state_path, default)

    def _save_state(self):
        self._state["ts"] = _now()
        _safe_json_dump(self._state_path, self._state)

    def _log_event(self, gesture: str, action: str, face: str, extra: Optional[Dict[str, Any]] = None):
        ev = {
            "ts": _now(),
            "gesture": gesture,
            "action": action,
            "face": face,
            "robot_state_before": self._state.get("robot_state", "UNKNOWN"),
        }
        if extra:
            ev.update(extra)
        _append_jsonl(self._log_path, ev)

    # ---------------- robot state helper ----------------
    def _need_support_stand(self, desired_action: str) -> bool:
        """
        Nếu robot đang SIT mà muốn MOVE/STAND => cần support_stand trước.
        """
        st = (self._state.get("robot_state") or "UNKNOWN").upper()
        if st == "SIT":
            if desired_action in ("STAND", "FORWARD", "TROT_FORWARD", "BACK", "TURN_LEFT", "TURN_RIGHT"):
                return True
        return False

    def _do_support_stand_if_needed(self, desired_action: str):
        if not self._need_support_stand(desired_action):
            return
        if self.boot_helper and hasattr(self.boot_helper, "support_stand"):
            print("[HandCommand] support_stand() because robot_state=SIT")
            try:
                self.boot_helper.support_stand(step=1, delay=0.03, pause_sec=0.6)
            except Exception as e:
                print("[HandCommand] support_stand error:", e)

    # ---------------- command dispatcher ----------------
    def _emit(self, gesture: str, action: str, face: str, bark: bool = False):
        now = _now()
        if now - self._last_emit_ts < float(self.cfg.cooldown_sec):
            return

        # hold gesture stable a little (anti jitter)
        if self._last_gesture != gesture:
            self._last_gesture = gesture
            self._gesture_since = now
            return
        if (now - self._gesture_since) < float(self.cfg.same_gesture_hold_sec):
            return

        self._last_emit_ts = now

        # check state safety
        self._do_support_stand_if_needed(action)

        # face first
        if face:
            self.set_face(face)

        # bark (if requested)
        if bark and self.on_bark:
            try:
                self.on_bark()
            except Exception:
                pass

        # send command
        if action and self.on_cmd:
            try:
                self.on_cmd(action)
            except Exception:
                pass

        # update state guess
        if action == "SIT":
            self._state["robot_state"] = "SIT"
        elif action == "STAND":
            self._state["robot_state"] = "STAND"
        elif action == "STOP":
            self._state["robot_state"] = "STOPPED"
        else:
            # forward/back/turn/trot
            self._state["robot_state"] = "MOVING"

        self._state["last_action"] = action
        self._state["last_face"] = face
        self._state["last_gesture"] = gesture
        self._save_state()

        self._log_event(gesture, action, face, extra={"bark": bool(bark)})

        print(f"[HandCommand] GESTURE={gesture} => ACTION={action} FACE={face} bark={bark}")

    # ---------------- camera ----------------
    def _open_cam(self):
        if self._cap is not None:
            return
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        cap = cv2.VideoCapture(self.cfg.cam_dev, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.cam_w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.cam_h))
        cap.set(cv2.CAP_PROP_FPS, int(self.cfg.cam_fps))
        self._cap = cap

    def _release_cam(self):
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    # ---------------- mediapipe helpers ----------------
    def _lm_xy(self, lm, w, h) -> Tuple[float, float]:
        return (float(lm.x) * w, float(lm.y) * h)

    def _hand_features(self, lms, w, h) -> Dict[str, Any]:
        """
        Return features from 21 landmarks:
          - finger up states
          - palm center
          - bbox
        """
        # indices
        WRIST = 0
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20

        INDEX_PIP = 6
        MIDDLE_PIP = 10
        RING_PIP = 14
        PINKY_PIP = 18

        pts = [self._lm_xy(lm, w, h) for lm in lms]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        palm = pts[WRIST]

        # simple "finger up": tip.y < pip.y  (camera coord: y down)
        def up(tip_i, pip_i):
            return pts[tip_i][1] < pts[pip_i][1]

        idx_up = up(INDEX_TIP, INDEX_PIP)
        mid_up = up(MIDDLE_TIP, MIDDLE_PIP)
        ring_up = up(RING_TIP, RING_PIP)
        pink_up = up(PINKY_TIP, PINKY_PIP)

        # thumb: use x-direction relative to index MCP to estimate extended
        # MCP index = 5
        thumb_tip = pts[THUMB_TIP]
        index_mcp = pts[5]
        thumb_ext = abs(thumb_tip[0] - index_mcp[0]) > (0.12 * (bbox[2] - bbox[0] + 1e-6))

        # fist: all fingers down (thumb may be near)
        fist = (not idx_up) and (not mid_up) and (not ring_up) and (not pink_up)

        # open palm: 4 fingers up
        open4 = idx_up and mid_up and ring_up and pink_up

        # thumb direction (left/right) based on thumb tip relative to wrist
        wrist = pts[WRIST]
        thumb_dir = "CENTER"
        dx = thumb_tip[0] - wrist[0]
        if abs(dx) > 0.10 * (bbox[2] - bbox[0] + 1e-6):
            thumb_dir = "RIGHT" if dx > 0 else "LEFT"

        return {
            "pts": pts,
            "bbox": bbox,
            "palm": palm,
            "idx_up": idx_up,
            "mid_up": mid_up,
            "ring_up": ring_up,
            "pink_up": pink_up,
            "thumb_ext": thumb_ext,
            "thumb_dir": thumb_dir,
            "fist": fist,
            "open4": open4,
        }

    def _detect_clap(self, hands_feats: List[Dict[str, Any]]) -> bool:
        """
        Clap = 2 hands present and distance between palms is very small,
        and just closed quickly recently.
        """
        if len(hands_feats) < 2:
            return False

        p1 = hands_feats[0]["palm"]
        p2 = hands_feats[1]["palm"]
        d = _dist(p1, p2)

        # normalize threshold by frame width (~640)
        # Clap close threshold
        close_thr = 90.0

        now = _now()

        if d < close_thr:
            # if recently also far then close -> clap
            if (now - self._last_clap_close_ts) > 0.25:
                # mark first close
                self._last_clap_close_ts = now
                return False
            # second detection within short window => clap confirmed
            return True

        # far: reset
        if d > 170.0:
            self._last_clap_close_ts = 0.0

        return False

    def _detect_beckon(self, feat: Dict[str, Any], prev_feat: Optional[Dict[str, Any]], w: int, h: int) -> bool:
        """
        "ngoắc lại gần bằng 4 ngón tay vẫy liên tục":
          - 4 fingers up (open4)
          - fingertip y changes back/forth between frames (wave)
        """
        if not feat["open4"]:
            return False
        if prev_feat is None:
            return False

        # track average fingertip y
        tips_idx = [8, 12, 16, 20]
        cur = np.mean([feat["pts"][i][1] for i in tips_idx])
        prv = np.mean([prev_feat["pts"][i][1] for i in tips_idx])

        dy = cur - prv
        # wave threshold
        return abs(dy) > 10.0

    def _palm_height(self, feat: Dict[str, Any], h: int) -> str:
        """
        return: HIGH / LOW / MID based on wrist/palm y
        """
        y = feat["palm"][1]
        if y < 0.30 * h:
            return "HIGH"
        if y > 0.70 * h:
            return "LOW"
        return "MID"

    # ---------------- main loop ----------------
    def _loop(self):
        self._open_cam()
        if self._cap is None or not self._cap.isOpened():
            print("[HandCommand] ERROR: camera not opened:", self.cfg.cam_dev)
            return

        prev_feat_one = None

        last_tick = 0.0
        min_dt = 1.0 / max(5.0, float(self.cfg.max_fps))

        while not self._stop.is_set():
            # limit fps
            tnow = _now()
            if (tnow - last_tick) < min_dt:
                time.sleep(0.002)
                continue
            last_tick = tnow

            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            # store last frame for dashboard use
            with self._last_frame_lock:
                self._last_frame = frame.copy()

            if not self.is_enabled():
                prev_feat_one = None
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = self._hands.process(rgb)
            hands_feats = []

            if res.multi_hand_landmarks:
                for hand_lms in res.multi_hand_landmarks:
                    feat = self._hand_features(hand_lms.landmark, w, h)
                    hands_feats.append(feat)

            # Optional overlay
            if self.cfg.draw_overlay and res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.multi_hand_landmarks[0], self._mp_hands.HAND_CONNECTIONS
                )

            # Clap (2 hands) highest priority
            if self._detect_clap(hands_feats):
                self._emit("CLAP", "BACK", "angry", bark=True)
                prev_feat_one = hands_feats[0] if hands_feats else None
                continue

            if not hands_feats:
                prev_feat_one = None
                continue

            # use first hand as primary
            feat0 = hands_feats[0]

            # 1) FIST => STOP (highest for single hand)
            if feat0["fist"]:
                self._emit("FIST", "STOP", "sleep")
                prev_feat_one = feat0
                continue

            # 2) Palm high/low => STAND/SIT
            ph = self._palm_height(feat0, h)
            if ph == "HIGH":
                self._emit("PALM_HIGH", "STAND", "suprise")
                prev_feat_one = feat0
                continue
            if ph == "LOW":
                self._emit("PALM_LOW", "SIT", "sad")
                prev_feat_one = feat0
                continue

            # 3) Index up only => TROT forward
            if feat0["idx_up"] and (not feat0["mid_up"]) and (not feat0["ring_up"]) and (not feat0["pink_up"]):
                self._emit("INDEX_UP", "TROT_FORWARD", "suprise")
                prev_feat_one = feat0
                continue

            # 4) Thumb left/right
            # require thumb extended, and other fingers down-ish to reduce false triggers
            if feat0["thumb_ext"] and (not feat0["idx_up"]) and (not feat0["mid_up"]) and (not feat0["ring_up"]) and (not feat0["pink_up"]):
                if feat0["thumb_dir"] == "RIGHT":
                    self._emit("THUMB_RIGHT", "TURN_RIGHT", "what_is_it")
                    prev_feat_one = feat0
                    continue
                if feat0["thumb_dir"] == "LEFT":
                    self._emit("THUMB_LEFT", "TURN_LEFT", "suprise")
                    prev_feat_one = feat0
                    continue

            # 5) Beckon (open4 + wave) => FORWARD
            if self._detect_beckon(feat0, prev_feat_one, w, h):
                self._emit("BECKON", "FORWARD", "suprise")
                prev_feat_one = feat0
                continue

            # default
            prev_feat_one = feat0

        # cleanup
        self._release_cam()
        try:
            self._hands.close()
        except Exception:
            pass
