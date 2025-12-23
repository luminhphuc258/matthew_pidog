#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Client PiDog (ActiveListenerV2) compatible with NEW NodeJS server response:
- If server returns play.type == "youtube":
    -> start systemd service robot-video-player (and pass youtube URL via env file)
    -> mark video playing so mic recording pauses (avoid self-record)
- If server returns audio_url (TTS/chat or iTunes mp3):
    -> stop video (if playing)
    -> set face "suprise" (and optional mouth animation handled by your face service)
    -> play audio (blocking)
- If bark (clap OR env noise):
    -> stop video
    -> set face "sad"
    -> play local tiengsua.wav (blocking)

IMPORTANT:
- This code assumes you already have:
    - face service listening UDP at 127.0.0.1:39393 (as you showed)
    - robot-video-player.service that reads /home/matthewlupi/matthew_pidog/robot_video_player.env
      with YOUTUBE_URL=... (and optionally VOLUME=...)
- Change paths below if different.
"""

import os
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import time
import json
import math
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import requests

try:
    import cv2
except Exception:
    cv2 = None

from robot_hat import Music
from motion_controller import MotionController


# ==========================
# Face (UDP) helpers
# ==========================
FACE_UDP_HOST = "127.0.0.1"
FACE_UDP_PORT = 39393

VALID_EMOS = {"love_eyes", "music", "what_is_it", "suprise", "sleep", "sad", "angry"}


def _udp_send(host: str, port: int, msg: str):
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(msg.encode("utf-8"), (host, port))
        s.close()
    except Exception:
        pass


def face_set_emo(emo: str):
    if emo not in VALID_EMOS:
        emo = "what_is_it"
    _udp_send(FACE_UDP_HOST, FACE_UDP_PORT, f"EMO {emo}")


def face_mouth(level01: float):
    # if your face service supports MOUTH 0..1 (it does)
    try:
        lv = max(0.0, min(1.0, float(level01)))
    except Exception:
        lv = 0.0
    _udp_send(FACE_UDP_HOST, FACE_UDP_PORT, f"MOUTH {lv:.3f}")


# ==========================
# Video service helpers
# ==========================
VIDEO_SERVICE_NAME = "robot-video-player.service"

# Env file for service to read.
# You said robot_video-player is in /home/matthewlupi/matthew_pidog
VIDEO_ENV_FILE = "/home/matthewlupi/matthew_pidog/robot_video_player.env"

# Optional: if your service supports these keys.
VIDEO_ENV_KEY_URL = "YOUTUBE_URL"
VIDEO_ENV_KEY_VOL = "VOLUME"

# If you want to force stop video when switching to audio
VIDEO_STOP_ON_AUDIO = True


def _write_env_kv(env_path: str, kv: Dict[str, str]):
    """
    Write simple KEY=VALUE lines (overwrites file).
    """
    lines = []
    for k, v in kv.items():
        k = str(k).strip()
        v = str(v).strip()
        # quote if contains spaces
        if " " in v or "\t" in v:
            v = '"' + v.replace('"', '\\"') + '"'
        lines.append(f"{k}={v}")
    tmp = env_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, env_path)


def video_is_active() -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "--quiet", VIDEO_SERVICE_NAME],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def video_start(youtube_url: str, volume: Optional[int] = None) -> bool:
    try:
        kv = {VIDEO_ENV_KEY_URL: youtube_url}
        if volume is not None:
            kv[VIDEO_ENV_KEY_VOL] = str(int(volume))
        _write_env_kv(VIDEO_ENV_FILE, kv)

        # restart to apply new url
        subprocess.run(
            ["systemctl", "restart", VIDEO_SERVICE_NAME],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception as e:
        print("[VIDEO] start error:", e)
        return False


def video_stop() -> bool:
    try:
        subprocess.run(
            ["systemctl", "stop", VIDEO_SERVICE_NAME],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception as e:
        print("[VIDEO] stop error:", e)
        return False


# ==========================
# Config
# ==========================
@dataclass
class ListenerCfg:
    # audio record
    mic_device: str = "default"
    sample_rate: int = 16000

    detect_chunk_sec: float = 0.6
    record_sec: float = 6.0

    # noise gate
    noise_calib_sec: float = 2.0
    gate_db_above_noise: float = 10.0
    min_rms_floor: float = 700.0

    # speech score
    speech_score_threshold: float = 0.62

    # clap detector
    clap_peak_ratio: float = 5.0
    clap_high_ratio: float = 0.12
    clap_zcr: float = 0.10

    # server
    server_url: str = "https://embeddedprogramming-healtheworldserver.up.railway.app/pi_upload_audio_v2"
    timeout_sec: float = 30.0

    # camera
    cam_dev: str = "/dev/video0"
    cam_w: int = 640
    cam_h: int = 480
    jpeg_quality: int = 80
    cam_warmup_frames: int = 6
    cam_backend: str = "v4l2"

    # memory
    memory_file: str = "robot_memory.jsonl"
    memory_max_items_send: int = 12

    # bark
    bark_wav: str = "tiengsua.wav"
    bark_times: int = 2

    # playback cooldown to avoid self-trigger
    playback_cooldown_sec: float = 0.8

    # speaker volume 0-100
    volume: int = 80

    # face behavior
    face_idle: str = "what_is_it"
    face_talk: str = "suprise"
    face_bark: str = "sad"

    # video
    video_default_volume: int = 80
    video_guard_start_sec: float = 0.8  # small guard before recording after starting video
    video_guard_stop_sec: float = 0.5   # guard after stopping video


class ActiveListenerV2:
    def __init__(self, cfg: ListenerCfg):
        self.cfg = cfg
        self._stop = False

        # audio playing flag (TTS/music)
        self._playing = False

        # video playing flag
        self._video_playing = False

        # cooldown
        self._cooldown_until = 0.0

        self.music = Music()
        try:
            self.music.music_set_volume(int(self.cfg.volume))
        except Exception:
            pass

        self._mem_path = Path(self.cfg.memory_file)
        self._bark_path = Path(self.cfg.bark_wav)

        self._noise_rms: Optional[float] = None

        # unlock speaker pin (if required)
        try:
            os.system("pinctrl set 12 op dh")
            time.sleep(0.1)
        except Exception:
            pass

        # set idle face
        face_set_emo(self.cfg.face_idle)

        # sync initial video status
        self._video_playing = video_is_active()

    def stop(self):
        self._stop = True

    # ---------- duration helper ----------
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

    # ---------- SAFE MUSIC PLAY (BLOCKING) ----------
    def _music_play_blocking(self, filepath: str, times: int = 1, talk_face: bool = False):
        """
        BLOCK playback and disable mic logic while playing.
        If talk_face=True: set suprise face before playing.
        """
        if VIDEO_STOP_ON_AUDIO:
            self._ensure_video_stopped()

        self._playing = True
        try:
            if talk_face:
                # show talk face and allow your face service to animate mouth itself.
                face_set_emo(self.cfg.face_talk)

            dur = self._get_audio_duration_sec(filepath)
            loops = max(1, int(times))

            # play
            try:
                self.music.music_set_volume(int(self.cfg.volume))
            except Exception:
                pass
            self.music.music_play(str(filepath), loops=loops)

            # best-effort block
            if dur is not None:
                time.sleep(dur * loops + 0.20)
            else:
                time.sleep(2.5 * loops)

        except Exception as e:
            print("[PLAY] error:", e)
        finally:
            self._playing = False
            self._cooldown_until = time.time() + float(self.cfg.playback_cooldown_sec)
            # return to idle face (unless video is playing)
            if not self._video_playing:
                face_set_emo(self.cfg.face_idle)

    def _ensure_video_stopped(self):
        """
        Stop video service and apply a short guard (cooldown) to avoid self-record.
        """
        if self._video_playing or video_is_active():
            print("[VIDEO] stopping ...")
            video_stop()
            self._video_playing = False
            self._cooldown_until = max(self._cooldown_until, time.time() + float(self.cfg.video_guard_stop_sec))

    def _start_youtube_video(self, url: str):
        """
        Start video service (restart) with url, set video playing flag and pause mic recording.
        """
        if not url:
            return
        # set a "music" face while video running (optional)
        face_set_emo("music")

        ok = video_start(url, volume=int(self.cfg.video_default_volume))
        self._video_playing = ok or video_is_active()
        # guard: stop recording immediately after starting
        self._cooldown_until = max(self._cooldown_until, time.time() + float(self.cfg.video_guard_start_sec))
        print("[VIDEO] start url:", url, "ok=", self._video_playing)

    def _bark(self):
        # bark => sad face + stop video + play local wav
        self._ensure_video_stopped()
        face_set_emo(self.cfg.face_bark)

        if not self._bark_path.exists():
            print(f"[WARN] bark file not found: {self._bark_path}")
            return
        self._music_play_blocking(str(self._bark_path), times=int(self.cfg.bark_times), talk_face=False)

    # ---------- MAIN ----------
    def run_forever(self):
        print("[ActiveListenerV2] start listening...")
        self._calibrate_noise_floor()

        while not self._stop:
            # update video status occasionally (cheap)
            if int(time.time()) % 3 == 0:
                self._video_playing = video_is_active()

            # ✅ mic off while playing audio OR video OR in cooldown
            if self._playing or self._video_playing or (time.time() < self._cooldown_until):
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

                # clap first
                if self._is_clap(feats):
                    print(f"[CLAP] rms={rms:.0f} pk/rms={feats['peak_ratio']:.1f} hi={feats['high_ratio']:.3f} zcr={feats['zcr']:.3f} -> BARK")
                    self._bark()
                    continue

                # speech check
                if feats["speech_score"] >= self.cfg.speech_score_threshold:
                    print(f"[SPEECH] rms={rms:.0f} score={feats['speech_score']:.2f} dbg={self._dbg_dict(feats)} -> record full")

                    full_wav = self._record_wav(seconds=self.cfg.record_sec)
                    if not full_wav:
                        continue

                    image_bytes = self._capture_jpeg_frame()
                    resp = self._send_to_server(full_wav, image_bytes=image_bytes)
                    if resp:
                        self._handle_server_reply(resp)

                    try:
                        os.remove(full_wav)
                    except Exception:
                        pass
                else:
                    # env noise => bark
                    print(f"[ENV] rms={rms:.0f} score={feats['speech_score']:.2f} dbg={self._dbg_dict(feats)} -> BARK")
                    self._bark()

            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    # ---------- NOISE FLOOR ----------
    def _calibrate_noise_floor(self):
        secs = max(1.0, float(self.cfg.noise_calib_sec))
        n_chunks = int(math.ceil(secs / max(0.2, self.cfg.detect_chunk_sec)))
        samples = []

        print(f"[CALIB] measuring noise floor for ~{secs:.1f}s ... keep quiet if possible")
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

        print(f"[CALIB] noise_rms≈{self._noise_rms:.0f}  gate=+{self.cfg.gate_db_above_noise:.1f}dB")

    def _passes_gate(self, rms: float) -> bool:
        if self._noise_rms is None:
            return rms >= float(self.cfg.min_rms_floor)
        thr = self._noise_rms * (10.0 ** (float(self.cfg.gate_db_above_noise) / 20.0))
        return rms >= thr

    # ---------- AUDIO RECORD ----------
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

    # ---------- FEATURE EXTRACT ----------
    def _extract_features(self, wav_path: str) -> Dict[str, float]:
        x = self._read_wav_pcm16(wav_path)
        if x is None or len(x) < 256:
            return {"rms": 0.0, "speech_score": 0.0, "flatness": 1.0, "speech_ratio": 0.0, "high_ratio": 0.0, "low_ratio": 0.0, "zcr": 0.0, "peak_ratio": 0.0}

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
        e_low = band_energy(50, 250)
        total = float(np.sum(ps))

        speech_ratio = e_speech / (total + 1e-9)
        high_ratio = e_high / (total + 1e-9)
        low_ratio = e_low / (total + 1e-9)

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
            "low_ratio": float(low_ratio),
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

    # ---------- CLAP DETECT ----------
    def _is_clap(self, feats: Dict[str, float]) -> bool:
        return (
            feats["peak_ratio"] >= float(self.cfg.clap_peak_ratio)
            and feats["high_ratio"] >= float(self.cfg.clap_high_ratio)
            and feats["zcr"] >= float(self.cfg.clap_zcr)
        )

    # ---------- CAMERA ----------
    def _capture_jpeg_frame(self) -> Optional[bytes]:
        if cv2 is None:
            return None

        cap = None
        try:
            backend = cv2.CAP_V4L2 if self.cfg.cam_backend.lower() == "v4l2" else 0
            cap = cv2.VideoCapture(self.cfg.cam_dev, backend)
            if not cap.isOpened():
                return None

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_h)

            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass

            frame = None
            for _ in range(max(2, int(self.cfg.cam_warmup_frames))):
                ok, fr = cap.read()
                if ok and fr is not None:
                    frame = fr

            if frame is None:
                return None

            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])
            if not ok2:
                return None
            return buf.tobytes()

        except Exception:
            return None
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    # ---------- MEMORY ----------
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
            print("[MEM] write error:", e)

    # ---------- SERVER ----------
    def _send_to_server(self, wav_path: str, image_bytes: Optional[bytes]) -> Optional[Dict[str, Any]]:
        mem = self._load_recent_memory()
        meta = {"ts": time.time(), "client": "pidog", "memory": mem}

        files = {"audio": ("audio.wav", open(wav_path, "rb"), "audio/wav")}
        if image_bytes:
            files["image"] = ("frame.jpg", image_bytes, "image/jpeg")

        data = {"meta": json.dumps(meta, ensure_ascii=False)}

        try:
            print("[HTTP] POST", self.cfg.server_url, "image=" + ("yes" if image_bytes else "no"))
            r = requests.post(self.cfg.server_url, files=files, data=data, timeout=self.cfg.timeout_sec)
            if r.status_code != 200:
                print("[HTTP] bad status:", r.status_code, r.text[:300])
                return None
            return r.json()
        except Exception as e:
            print("[HTTP] error:", e)
            return None
        finally:
            try:
                files["audio"][1].close()
            except Exception:
                pass

    def _download(self, url: str, dst: str) -> bool:
        try:
            rr = requests.get(url, timeout=45)
            if rr.status_code != 200:
                return False
            with open(dst, "wb") as f:
                f.write(rr.content)
            return True
        except Exception:
            return False

    # ---------- NEW reply handling ----------
    def _handle_server_reply(self, resp: Dict[str, Any]):
        transcript = (resp.get("transcript") or "").strip()
        label = (resp.get("label") or "unknown").strip()
        reply_text = (resp.get("reply_text") or "").strip()
        audio_url = resp.get("audio_url")

        # ✅ NEW: play field for youtube
        play = resp.get("play") or {}
        play_type = str(play.get("type") or "").strip().lower()
        play_url = str(play.get("url") or play.get("youtube_url") or "").strip()

        print("[SERVER] label=", label)
        if transcript:
            print("[USER ]", transcript)
        if reply_text:
            print("[BOT  ]", reply_text)
        if audio_url:
            print("[AUDIO]", audio_url)
        if play_type:
            print("[PLAY ]", play_type, play_url)

        self._append_memory({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": transcript,
            "label": label,
            "reply_text": reply_text,
            "audio_url": audio_url,
            "play": play,
        })

        # clap label => bark local
        if label.lower() == "clap":
            print("[CLAP->BARK] local bark")
            self._bark()
            return

        # ✅ If server tells youtube playback
        if play_type == "youtube" and play_url:
            # start video service, pause mic recording while playing
            self._start_youtube_video(play_url)
            return

        # ✅ If server returns audio_url: stop video then talk face + play audio
        if not audio_url:
            return

        tmpdir = tempfile.mkdtemp(prefix="al2_play_")
        local = os.path.join(tmpdir, "reply.mp3")
        try:
            # stop video before speaking (requested)
            if VIDEO_STOP_ON_AUDIO:
                self._ensure_video_stopped()

            if not self._download(audio_url, local):
                print("[PLAY] download failed:", audio_url)
                return

            # talk face + let your face service do mouth animation (already in your service)
            self._music_play_blocking(local, times=1, talk_face=True)

        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


def main():
    POSE_FILE = Path(__file__).resolve().parent / "pidog_pose_config.txt"
    mc = MotionController(pose_file=POSE_FILE)

    print("[1] BOOT")
    mc.boot()
    time.sleep(0.8)

    cfg = ListenerCfg(
        mic_device="default",
        server_url="https://embeddedprogramming-healtheworldserver.up.railway.app/pi_upload_audio_v2",
        bark_wav="tiengsua.wav",
        bark_times=2,
        cam_dev="/dev/video0",
        cam_backend="v4l2",
        gate_db_above_noise=10.0,
        speech_score_threshold=0.62,
        playback_cooldown_sec=0.8,
        volume=80,
        face_idle="what_is_it",
        face_talk="suprise",
        face_bark="sad",
        video_default_volume=80,
    )

    al = ActiveListenerV2(cfg)

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
