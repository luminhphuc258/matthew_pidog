#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import time
import json
import math
import shutil
import tempfile
import subprocess
import threading
import socket
import ssl
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import requests

# CV2 vẫn có thể giữ, nhưng với snapshot_file thì ActiveListener sẽ không mở camera nữa
try:
    import cv2
except Exception:
    cv2 = None

import paho.mqtt.client as mqtt

from robot_hat import Music
from motion_controller import MotionController


# =========================
# Face UDP helper (your pygame face service)
# =========================
FACE_UDP_HOST = "127.0.0.1"
FACE_UDP_PORT = 39393
VALID_EMOS = {"love_eyes", "music", "what_is_it", "suprise", "sleep", "sad", "angry"}  # match your face code


def _udp_send(msg: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(msg.encode("utf-8", "ignore"), (FACE_UDP_HOST, FACE_UDP_PORT))
        s.close()
    except Exception:
        pass


def set_face(emo: str):
    if emo in VALID_EMOS:
        _udp_send(f"EMO {emo}")


def set_mouth(level01: float):
    level01 = max(0.0, min(1.0, float(level01)))
    _udp_send(f"MOUTH {level01:.3f}")


# =========================
# MQTT Gesture topics (subscribe)
# =========================
MQTT_HOST = "rfff7184.ala.us-east-1.emqxsl.com"
MQTT_PORT = 8883
MQTT_USER = "robot_matthew"
MQTT_PASS = "29061992abCD!yesokmen"

# topics sent by your gesture-detect service
GESTURE_TOPICS = {
    "STOPMUSIC": "/robot/gesture/stopmusic",
    "STANDUP":   "robot/gesture/standup",
    "SIT":       "robot/gesture/sit",
    "MOVELEFT":  "robot/gesture/moveleft",
    "MOVERIGHT": "robot/moveright",
    "STOP":      "/robot/gesture/stop",
}


# =========================
# Dummy JPEG (1x1) fallback
# - dùng khi snapshot chưa có / đọc lỗi
# =========================
_DUMMY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAALCAAaABoBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAHf/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPwB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwB//9k="
)
DUMMY_JPEG_BYTES = base64.b64decode(_DUMMY_JPEG_B64)


@dataclass
class ListenerCfg:
    # audio input
    mic_device: str = "default"
    sample_rate: int = 16000

    detect_chunk_sec: float = 0.6
    record_sec: float = 6.0

    # ===== auto noise gate (LOW like your old) =====
    noise_calib_sec: float = 1.2
    gate_db_above_noise: float = 4.0
    min_rms_floor: float = 200.0

    # speech score (LOW)
    speech_score_threshold: float = 0.55

    # clap detector
    clap_peak_ratio: float = 5.0
    clap_high_ratio: float = 0.12
    clap_zcr: float = 0.10

    # server (still used for STT/TTS)
    server_url: str = "https://embeddedprogramming-healtheworldserver.up.railway.app/pi_upload_audio_v2"
    timeout_sec: float = 30.0

    # camera (legacy - nếu snapshot_file có thì ActiveListener sẽ không mở cam)
    cam_dev: str = "/dev/video0"
    cam_w: int = 640
    cam_h: int = 480
    jpeg_quality: int = 80
    cam_warmup_frames: int = 6
    cam_backend: str = "v4l2"

    # snapshot (NEW) - đọc ảnh từ gesture service
    snapshot_file: Optional[str] = "/tmp/gesture_latest.jpg"
    snapshot_max_age_sec: float = 3.0  # nếu file quá cũ => fallback dummy

    # server requires image
    always_send_image: bool = True

    # memory
    memory_file: str = "robot_memory.jsonl"
    memory_max_items_send: int = 12

    # bark
    bark_wav: str = "tiengsua.wav"
    bark_times: int = 2

    # ✅ waiting message while waiting HTTP response
    waiting_wav: str = "waitingmessage.wav"
    waiting_enable: bool = True

    # cooldown after playback to avoid self-trigger
    playback_cooldown_sec: float = 0.7

    # volume
    volume: int = 80

    # mqtt enable
    mqtt_enable: bool = True


class GestureMqttSubscriber:
    """
    Subscribe gesture topics and call callbacks.
    TLS insecure: no CA.
    """
    def __init__(self, on_cmd):
        self.on_cmd = on_cmd
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)

        # TLS insecure (no verify)
        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self._connected = False
        self._topic_to_cmd = {v: k for k, v in GESTURE_TOPICS.items()}

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = (rc == 0)
        print(f"[MQTT] connected rc={rc}", flush=True)
        if rc == 0:
            for tp in self._topic_to_cmd.keys():
                try:
                    client.subscribe(tp, qos=0)
                    print(f"[MQTT] subscribe {tp}", flush=True)
                except Exception as e:
                    print("[MQTT] subscribe error:", e, flush=True)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[MQTT] disconnected rc={rc}", flush=True)

    def _on_message(self, client, userdata, msg):
        try:
            topic = (msg.topic or "").strip()
            cmd = self._topic_to_cmd.get(topic)
            if not cmd:
                return
            print(f"[MQTT] recv {topic} -> {cmd}", flush=True)
            self.on_cmd(cmd)
        except Exception as e:
            print("[MQTT] on_message error:", e, flush=True)

    def start(self):
        self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.client.loop_start()

    def stop(self):
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass


class ActiveListenerV2:
    def __init__(self, cfg: ListenerCfg, motion: Optional[MotionController] = None):
        self.cfg = cfg
        self.motion = motion

        self._stop = False

        # playback state
        self._playing_audio = False
        self._cooldown_until = 0.0

        # allow STOPMUSIC/STOP to interrupt playback immediately
        self._playback_stop_evt = threading.Event()

        # motion command queue (from MQTT)
        self._cmd_lock = threading.Lock()
        self._pending_cmds: List[str] = []
        self._last_cmd_ts: Dict[str, float] = {}
        self._cmd_cooldown = 0.25

        self.music = Music()
        try:
            self.music.music_set_volume(int(self.cfg.volume))
        except Exception:
            pass

        self._mem_path = Path(self.cfg.memory_file)
        self._bark_path = Path(self.cfg.bark_wav)
        self._waiting_path = Path(self.cfg.waiting_wav)

        self._noise_rms: Optional[float] = None

        # unlock speaker pin if needed (Robot HAT)
        try:
            os.system("pinctrl set 12 op dh")
            time.sleep(0.1)
        except Exception:
            pass

        # mqtt subscriber
        self._mqtt = None
        if self.cfg.mqtt_enable:
            self._mqtt = GestureMqttSubscriber(self._on_gesture_cmd)
            self._mqtt.start()

        # default face
        set_face("what_is_it")
        set_mouth(0.0)

    def stop(self):
        self._stop = True
        try:
            if self._mqtt:
                self._mqtt.stop()
        except Exception:
            pass

    # ----------------------------
    # MQTT gesture handling
    # ----------------------------
    def _on_gesture_cmd(self, cmd: str):
        now = time.time()
        with self._cmd_lock:
            last = self._last_cmd_ts.get(cmd, 0.0)
            if (now - last) < self._cmd_cooldown:
                return
            self._last_cmd_ts[cmd] = now

        if cmd in ("STOPMUSIC", "STOP"):
            print(f"[GESTURE] {cmd} -> stop audio + resume listening", flush=True)
            self.request_stopmusic()
            return

        with self._cmd_lock:
            self._pending_cmds.append(cmd)

    def request_stopmusic(self):
        self._playback_stop_evt.set()
        self._stop_audio_playback()
        self._playing_audio = False
        self._cooldown_until = time.time() + 0.05
        set_face("what_is_it")
        set_mouth(0.0)

    def _pop_pending_cmds(self) -> List[str]:
        with self._cmd_lock:
            cmds = self._pending_cmds[:]
            self._pending_cmds.clear()
        return cmds

    def _handle_motion_cmd(self, cmd: str):
        if not self.motion:
            return

        action = None
        if cmd == "STANDUP":
            action = "STAND"
        elif cmd == "SIT":
            action = "SIT"
        elif cmd == "MOVELEFT":
            action = "TURN_LEFT"
        elif cmd == "MOVERIGHT":
            action = "TURN_RIGHT"
        elif cmd == "STOP":
            action = "STOP"

        if not action:
            return

        print(f"[MOTION] {cmd} -> {action}", flush=True)

        try:
            if hasattr(self.motion, "move"):
                self.motion.move(action)
                return
            if hasattr(self.motion, "command"):
                self.motion.command(action)
                return
            if action == "STAND" and hasattr(self.motion, "stand"):
                self.motion.stand(); return
            if action == "SIT" and hasattr(self.motion, "sit"):
                self.motion.sit(); return
            if action == "TURN_LEFT" and hasattr(self.motion, "turn_left"):
                self.motion.turn_left(); return
            if action == "TURN_RIGHT" and hasattr(self.motion, "turn_right"):
                self.motion.turn_right(); return
            if action == "STOP" and hasattr(self.motion, "stop"):
                self.motion.stop(); return
        except Exception as e:
            print("[MOTION] error:", e, flush=True)

    # ----------------------------
    # Audio playback + lipsync
    # ----------------------------
    def _stop_audio_playback(self):
        try:
            if hasattr(self.music, "music_stop"):
                self.music.music_stop()
        except Exception:
            pass

    def _get_audio_duration_sec(self, filepath: str) -> Optional[float]:
        p = str(filepath)
        if p.lower().endswith(".wav"):
            try:
                import wave
                with wave.open(p, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return float(frames) / float(rate)
            except Exception:
                pass

        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", p],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False
            )
            s = (r.stdout or "").strip()
            if s:
                return float(s)
        except Exception:
            pass
        return None

    def _lipsync_worker(self, audio_path: str, stop_evt: threading.Event):
        try:
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", audio_path,
                "-ac", "1", "-ar", "16000",
                "-f", "s16le", "pipe:1"
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            chunk_samples = int(16000 * 0.05)
            chunk_bytes = chunk_samples * 2
            while not stop_evt.is_set():
                buf = p.stdout.read(chunk_bytes) if p.stdout else b""
                if not buf:
                    break
                x = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(x * x) + 1e-9))
                level = min(1.0, max(0.0, (rms / 4000.0)))
                set_mouth(level)
            try:
                p.terminate()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            set_mouth(0.0)

    def _music_play_blocking_with_lipsync(self, filepath: str, times: int = 1):
        self._playing_audio = True
        self._playback_stop_evt.clear()

        stop_evt = threading.Event()
        th = None
        try:
            dur = self._get_audio_duration_sec(filepath)
            loops = max(1, int(times))

            set_face("suprise")
            set_mouth(0.0)

            th = threading.Thread(target=self._lipsync_worker, args=(filepath, stop_evt), daemon=True)
            th.start()

            self.music.music_play(str(filepath), loops=loops)

            if dur is None:
                dur = 2.5
            total = dur * loops + 0.15

            t_end = time.time() + total
            while time.time() < t_end:
                if self._playback_stop_evt.is_set():
                    print("[PLAY] interrupted by STOP/STOPMUSIC", flush=True)
                    self._stop_audio_playback()
                    break
                time.sleep(0.05)

        except Exception as e:
            print("[PLAY] error:", e, flush=True)
        finally:
            stop_evt.set()
            try:
                if th:
                    th.join(timeout=0.3)
            except Exception:
                pass

            self._playing_audio = False
            set_mouth(0.0)
            set_face("what_is_it")

            if self._playback_stop_evt.is_set():
                self._cooldown_until = time.time() + 0.05
            else:
                self._cooldown_until = time.time() + float(self.cfg.playback_cooldown_sec)

    def _bark(self):
        if not self._bark_path.exists():
            print(f"[WARN] bark file not found: {self._bark_path}", flush=True)
            return

        self._playing_audio = True
        self._playback_stop_evt.clear()

        set_face("sad")
        set_mouth(0.0)

        try:
            self.music.music_play(str(self._bark_path), loops=max(1, int(self.cfg.bark_times)))
            dur = self._get_audio_duration_sec(str(self._bark_path))
            if dur is None:
                dur = 2.2
            total = dur * max(1, int(self.cfg.bark_times)) + 0.15

            t_end = time.time() + total
            while time.time() < t_end:
                if self._playback_stop_evt.is_set():
                    print("[BARK] interrupted by STOP/STOPMUSIC", flush=True)
                    self._stop_audio_playback()
                    break
                time.sleep(0.05)

        finally:
            self._playing_audio = False
            set_face("what_is_it")
            set_mouth(0.0)

            if self._playback_stop_evt.is_set():
                self._cooldown_until = time.time() + 0.05
            else:
                self._cooldown_until = time.time() + float(self.cfg.playback_cooldown_sec)

    # ----------------------------
    # ✅ Waiting message loop
    # ----------------------------
    def _waiting_message_loop(self, stop_evt: threading.Event):
        """
        Play waitingmessage.wav repeatedly while waiting server response.
        Stops immediately when stop_evt set OR STOP/STOPMUSIC gesture triggered.
        """
        p = self._waiting_path
        if not self.cfg.waiting_enable:
            return
        if not p.exists():
            print(f"[WAIT] waiting file not found: {p}", flush=True)
            return

        dur = self._get_audio_duration_sec(str(p))
        if dur is None or dur <= 0.05:
            dur = 2.0

        print("[WAIT] start waitingmessage loop...", flush=True)
        self._playing_audio = True
        try:
            while (not stop_evt.is_set()) and (not self._stop):
                if self._playback_stop_evt.is_set():
                    # STOP/STOPMUSIC interrupt
                    self._stop_audio_playback()
                    break

                # play once
                try:
                    self.music.music_play(str(p), loops=1)
                except Exception as e:
                    print("[WAIT] play error:", e, flush=True)
                    break

                t_end = time.time() + float(dur) + 0.05
                while time.time() < t_end:
                    if stop_evt.is_set() or self._stop or self._playback_stop_evt.is_set():
                        try:
                            self._stop_audio_playback()
                        except Exception:
                            pass
                        break
                    time.sleep(0.05)

                # loop again if still waiting
        finally:
            try:
                self._stop_audio_playback()
            except Exception:
                pass
            self._playing_audio = False
            print("[WAIT] stop waitingmessage loop", flush=True)

    # ----------------------------
    # Main loop
    # ----------------------------
    def run_forever(self):
        print("[ActiveListenerV2] start listening...", flush=True)
        self._calibrate_noise_floor()

        while not self._stop:
            cmds = self._pop_pending_cmds()
            for c in cmds:
                self._handle_motion_cmd(c)

            if self._playing_audio or (time.time() < self._cooldown_until):
                time.sleep(0.05)
                continue

            wav_path = self._record_wav(seconds=self.cfg.detect_chunk_sec)
            if not wav_path:
                time.sleep(0.08)
                continue

            try:
                feats = self._extract_features(wav_path)
                rms = feats["rms"]

                if not self._passes_gate(rms):
                    continue

                if self._is_clap(feats):
                    print(f"[CLAP] rms={rms:.0f} peak/rms={feats['peak_ratio']:.1f} hi={feats['high_ratio']:.3f} zcr={feats['zcr']:.3f} -> BARK", flush=True)
                    self._bark()
                    continue

                if feats["speech_score"] >= float(self.cfg.speech_score_threshold):
                    print(f"[SPEECH] rms={rms:.0f} score={feats['speech_score']:.2f} dbg={self._dbg_dict(feats)} -> record full", flush=True)

                    full_wav = self._record_wav(seconds=self.cfg.record_sec)
                    if not full_wav:
                        continue

                    # ✅ đọc ảnh snapshot từ gesture service
                    image_bytes = self._get_image_bytes_for_request()

                    # ✅ START waiting sound while HTTP request is pending
                    wait_stop = threading.Event()
                    wait_th = None
                    if self.cfg.waiting_enable and self._waiting_path.exists():
                        wait_th = threading.Thread(
                            target=self._waiting_message_loop,
                            args=(wait_stop,),
                            daemon=True
                        )
                        wait_th.start()

                    try:
                        resp = self._send_to_server(full_wav, image_bytes=image_bytes)
                    finally:
                        # stop waiting loop immediately (even on error/timeout)
                        wait_stop.set()
                        try:
                            self._stop_audio_playback()
                        except Exception:
                            pass
                        if wait_th:
                            try:
                                wait_th.join(timeout=1.2)
                            except Exception:
                                pass
                        # ensure not stuck in playing state
                        self._playing_audio = False

                    if resp:
                        self._handle_server_reply(resp)

                    try:
                        os.remove(full_wav)
                    except Exception:
                        pass
                else:
                    print(f"[ENV] rms={rms:.0f} score={feats['speech_score']:.2f} dbg={self._dbg_dict(feats)} -> BARK", flush=True)
                    self._bark()

            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    # ----------------------------
    # Snapshot image (READ FROM FILE)
    # ----------------------------
    def _get_image_bytes_for_request(self) -> Optional[bytes]:
        """
        Ưu tiên snapshot_file (gesture service xuất ảnh ra /tmp).
        - Nếu file không có / quá cũ / lỗi đọc => fallback dummy.
        """
        # 1) snapshot file first
        if self.cfg.snapshot_file:
            try:
                p = Path(self.cfg.snapshot_file)
                if p.exists() and p.stat().st_size > 500:
                    age = time.time() - p.stat().st_mtime
                    if age <= float(self.cfg.snapshot_max_age_sec):
                        return p.read_bytes()
                    else:
                        pass
            except Exception:
                pass

        # 2) fallback dummy
        if bool(self.cfg.always_send_image):
            return DUMMY_JPEG_BYTES

        return None

    # ----------------------------
    # Noise floor / gate
    # ----------------------------
    def _calibrate_noise_floor(self):
        secs = max(0.6, float(self.cfg.noise_calib_sec))
        n_chunks = int(math.ceil(secs / max(0.2, self.cfg.detect_chunk_sec)))
        samples = []

        print(f"[CALIB] measuring noise floor for ~{secs:.1f}s ... keep quiet if possible", flush=True)
        for _ in range(max(2, n_chunks)):
            if self._stop:
                break
            p = self._record_wav(seconds=self.cfg.detect_chunk_sec)
            if not p:
                continue
            try:
                x = self._read_wav_pcm16(p)
                if x is None or len(x) < 256:
                    continue
                rms = float(np.sqrt(np.mean(x * x) + 1e-9))
                samples.append(rms)
            finally:
                try:
                    os.remove(p)
                except Exception:
                    pass

        if not samples:
            self._noise_rms = float(self.cfg.min_rms_floor)
        else:
            med = float(np.median(np.array(samples)))
            self._noise_rms = max(float(self.cfg.min_rms_floor), med)

        print(f"[CALIB] noise_rms≈{self._noise_rms:.0f}  gate=+{self.cfg.gate_db_above_noise:.1f}dB", flush=True)

    def _passes_gate(self, rms: float) -> bool:
        floor = float(self.cfg.min_rms_floor)
        if self._noise_rms is None:
            return rms >= floor
        thr = self._noise_rms * (10.0 ** (float(self.cfg.gate_db_above_noise) / 20.0))
        thr = max(floor, thr)
        return rms >= thr

    # ----------------------------
    # Record wav
    # ----------------------------
    def _record_wav(self, seconds: float) -> Optional[str]:
        if seconds <= 0:
            return None

        tmp = tempfile.NamedTemporaryFile(prefix="al2_", suffix=".wav", delete=False)
        tmp.close()
        out = tmp.name

        cmd = [
            "arecord",
            "-D", self.cfg.mic_device,
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(self.cfg.sample_rate),
            "-d", str(max(1, int(math.ceil(seconds)))),
            out
        ]

        try:
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if p.returncode != 0 or (not Path(out).exists()) or Path(out).stat().st_size < 1200:
                try:
                    os.remove(out)
                except Exception:
                    pass
                return None
            return out
        except Exception:
            try:
                os.remove(out)
            except Exception:
                pass
            return None

    def _read_wav_pcm16(self, wav_path: str) -> Optional[np.ndarray]:
        import wave
        try:
            with wave.open(wav_path, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    return None
                raw = wf.readframes(wf.getnframes())
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        except Exception:
            return None

    # ----------------------------
    # Feature extract
    # ----------------------------
    def _extract_features(self, wav_path: str) -> Dict[str, float]:
        x = self._read_wav_pcm16(wav_path)
        if x is None or len(x) < 256:
            return {
                "rms": 0.0, "speech_score": 0.0, "flatness": 1.0, "speech_ratio": 0.0,
                "high_ratio": 0.0, "low_ratio": 0.0, "zcr": 0.0, "peak_ratio": 0.0
            }

        rms = float(np.sqrt(np.mean(x * x) + 1e-9))
        peak = float(np.max(np.abs(x)) + 1e-9)
        peak_ratio = float(peak / (rms + 1e-9))

        y = x / (np.max(np.abs(x)) + 1e-9)
        zc = float(np.mean(y[1:] * y[:-1] < 0.0))

        sr = self.cfg.sample_rate
        n = int(2 ** int(np.ceil(np.log2(len(y)))))
        yf = np.fft.rfft(y, n=n)
        ps = (np.abs(yf) ** 2) + 1e-12
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        flatness = float(np.exp(np.mean(np.log(ps))) / np.mean(ps))

        def band_energy(f1, f2):
            m = (freqs >= f1) & (freqs <= f2)
            return float(np.sum(ps[m]))

        e_speech = band_energy(300, 3400)
        e_high = band_energy(5000, 7500)
        total = float(np.sum(ps))

        speech_ratio = e_speech / (total + 1e-9)
        high_ratio = e_high / (total + 1e-9)

        score = 0.0
        score += (1.0 - min(1.0, flatness * 3.0)) * 0.40
        score += min(1.0, speech_ratio * 3.2) * 0.50
        score += (1.0 - min(1.0, high_ratio * 6.0)) * 0.10
        score = float(max(0.0, min(1.0, score)))

        return {
            "rms": rms,
            "speech_score": score,
            "flatness": float(flatness),
            "speech_ratio": float(speech_ratio),
            "high_ratio": float(high_ratio),
            "low_ratio": 0.0,
            "zcr": float(zc),
            "peak_ratio": peak_ratio,
        }

    def _dbg_dict(self, feats: Dict[str, float]) -> Dict[str, float]:
        return {
            "flat": round(feats["flatness"], 4),
            "sp": round(feats["speech_ratio"], 4),
            "hi": round(feats["high_ratio"], 4),
            "zcr": round(feats["zcr"], 4),
            "pk": round(feats["peak_ratio"], 2),
            "noise": round(float(self._noise_rms or 0.0), 1),
        }

    def _is_clap(self, feats: Dict[str, float]) -> bool:
        return (
            feats["peak_ratio"] >= float(self.cfg.clap_peak_ratio)
            and feats["high_ratio"] >= float(self.cfg.clap_high_ratio)
            and feats["zcr"] >= float(self.cfg.clap_zcr)
        )

    # ----------------------------
    # Memory
    # ----------------------------
    def _load_recent_memory(self) -> List[Dict[str, Any]]:
        if not self._mem_path.exists():
            return []
        items = []
        try:
            with self._mem_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            return []
        return items[-self.cfg.memory_max_items_send:]

    def _append_memory(self, entry: Dict[str, Any]):
        try:
            with self._mem_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print("[MEM] write error:", e, flush=True)

    # ----------------------------
    # Server
    # ----------------------------
    def _send_to_server(self, wav_path: str, image_bytes: Optional[bytes]) -> Optional[Dict[str, Any]]:
        mem = self._load_recent_memory()
        meta = {"ts": time.time(), "client": "pidog", "memory": mem}

        data = {"meta": json.dumps(meta, ensure_ascii=False)}

        audio_f = None
        try:
            audio_f = open(wav_path, "rb")
            files = {
                "audio": ("audio.wav", audio_f, "audio/wav"),
            }

            if self.cfg.always_send_image:
                if not image_bytes:
                    image_bytes = DUMMY_JPEG_BYTES
                files["image"] = ("frame.jpg", image_bytes, "image/jpeg")
            else:
                if image_bytes:
                    files["image"] = ("frame.jpg", image_bytes, "image/jpeg")

            is_dummy = ("image" in files and files["image"][1] == DUMMY_JPEG_BYTES)
            print("[HTTP] POST", self.cfg.server_url,
                  "image=" + ("yes" if ("image" in files) else "no"),
                  "dummy=" + ("yes" if is_dummy else "no"),
                  flush=True)

            r = requests.post(self.cfg.server_url, files=files, data=data, timeout=self.cfg.timeout_sec)
            if r.status_code != 200:
                print("[HTTP] bad status:", r.status_code, r.text[:300], flush=True)
                return None
            return r.json()

        except Exception as e:
            print("[HTTP] error:", e, flush=True)
            return None
        finally:
            try:
                if audio_f:
                    audio_f.close()
            except Exception:
                pass

    def _download(self, url: str, dst: str) -> bool:
        try:
            rr = requests.get(url, timeout=30)
            if rr.status_code != 200:
                return False
            with open(dst, "wb") as f:
                f.write(rr.content)
            return True
        except Exception:
            return False

    # ----------------------------
    # Handle server reply (AUDIO-ONLY)
    # ----------------------------
    def _handle_server_reply(self, resp: Dict[str, Any]):
        transcript = (resp.get("transcript") or resp.get("text") or "").strip()
        label = (resp.get("label") or "unknown").strip()
        reply_text = (resp.get("reply_text") or "").strip()
        audio_url = resp.get("audio_url")

        print("[SERVER] label=", label, flush=True)
        print("[USER ]", transcript, flush=True)
        if reply_text:
            print("[BOT  ]", reply_text, flush=True)
        print("[AUDIO]", audio_url, flush=True)

        self._append_memory({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": transcript,
            "label": label,
            "reply_text": reply_text,
            "audio_url": audio_url,
        })

        # NOTE: stop nhạc không nhận từ server nữa, mà qua MQTT STOP/STOPMUSIC
        if audio_url:
            tmpdir = tempfile.mkdtemp(prefix="al2_play_")
            local = os.path.join(tmpdir, "reply.mp3")
            try:
                if not self._download(audio_url, local):
                    print("[PLAY] download failed:", audio_url, flush=True)
                    return
                if label.strip().lower() == "nhac":
                    set_face("music")
                self._music_play_blocking_with_lipsync(local, times=1)
            finally:
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass


def main():
    POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"
    mc = MotionController(pose_file=POSE_FILE)

    print("[1] BOOT", flush=True)
    mc.boot()
    time.sleep(0.8)

    cfg = ListenerCfg(
        mic_device="default",
        server_url="https://embeddedprogramming-healtheworldserver.up.railway.app/pi_upload_audio_v2",
        bark_wav="tiengsua.wav",
        bark_times=2,

        # ✅ waiting message
        waiting_wav="waitingmessage.wav",
        waiting_enable=True,

        # ✅ snapshot from gesture service
        snapshot_file="/tmp/gesture_latest.jpg",
        snapshot_max_age_sec=3.0,
        always_send_image=True,

        # legacy cam fields (không dùng nếu snapshot_file ok)
        cam_dev="/dev/video0",
        cam_backend="v4l2",

        noise_calib_sec=1.2,
        gate_db_above_noise=4.0,
        min_rms_floor=200.0,
        speech_score_threshold=0.55,

        playback_cooldown_sec=0.7,
        volume=80,
        mqtt_enable=True,
    )

    al = ActiveListenerV2(cfg, motion=mc)

    try:
        al.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        al.stop()
        try:
            mc.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
