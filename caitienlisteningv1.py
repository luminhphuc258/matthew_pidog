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
import socket
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


# ============================================================
# UDP FACE CLIENT (talk to your pygame face service)
#   - service listens on 127.0.0.1:39393
#   - commands supported:
#       "EMO suprise"
#       "EMO sad"
#       "MOUTH 0.35"
# ============================================================
class FaceUdpClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 39393):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def emo(self, name: str):
        # your face service uses: suprise (typo) not surprise
        cmd = f"EMO {name}".encode("utf-8", "ignore")
        try:
            self.sock.sendto(cmd, self.addr)
        except Exception:
            pass

    def mouth(self, level01: float):
        v = max(0.0, min(1.0, float(level01)))
        cmd = f"MOUTH {v:.2f}".encode("utf-8", "ignore")
        try:
            self.sock.sendto(cmd, self.addr)
        except Exception:
            pass


# ============================================================
# Robot Video Player Controller
#   Assumption: your robot-video-player service exposes local HTTP:
#     POST http://127.0.0.1:9900/play  {"url": "..."}
#     POST http://127.0.0.1:9900/stop  {}
#     POST http://127.0.0.1:9900/face  {}   (switch back to face mode)
#
# If your endpoints are different, just change paths below.
# ============================================================
class VideoPlayerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:9900"):
        self.base = base_url.rstrip("/")
        self._mode = "face"   # best-effort state: face | video

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 2.0) -> bool:
        url = self.base + path
        try:
            r = requests.post(url, json=(payload or {}), timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def play_youtube(self, player_url: str) -> bool:
        ok = self._post("/play", {"url": player_url})
        if ok:
            self._mode = "video"
        return ok

    def stop_video(self) -> bool:
        ok = self._post("/stop", {})
        if ok:
            self._mode = "face"
        return ok

    def show_face(self) -> bool:
        ok = self._post("/face", {})
        if ok:
            self._mode = "face"
        return ok

    @property
    def mode(self) -> str:
        return self._mode


@dataclass
class ListenerCfg:
    # audio input
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

    # cooldown to avoid self-trigger
    playback_cooldown_sec: float = 0.7

    # speaker volume (0-100)
    volume: int = 80

    # local services
    face_udp_host: str = "127.0.0.1"
    face_udp_port: int = 39393
    video_service_url: str = "http://127.0.0.1:9900"

    # while playing youtube, disable mic this long if server provides no duration
    youtube_default_block_sec: float = 240.0  # 4 minutes default


class ActiveListenerMain:
    def __init__(self, cfg: ListenerCfg):
        self.cfg = cfg
        self._stop = False

        self.music = Music()
        try:
            self.music.music_set_volume(int(self.cfg.volume))
        except Exception:
            pass

        self.face = FaceUdpClient(cfg.face_udp_host, cfg.face_udp_port)
        self.video = VideoPlayerClient(cfg.video_service_url)

        self._mem_path = Path(self.cfg.memory_file)
        self._bark_path = Path(self.cfg.bark_wav)

        self._noise_rms: Optional[float] = None

        self._playing_audio = False
        self._block_until = 0.0  # for cooldown / youtube block

        # unlock speaker pin if needed
        try:
            os.system("pinctrl set 12 op dh")
            time.sleep(0.1)
        except Exception:
            pass

    def stop(self):
        self._stop = True

    # ----------------------------
    # Utilities: duration + lipsync envelope
    # ----------------------------
    def _ffprobe_duration(self, filepath: str) -> Optional[float]:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", filepath],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False
            )
            s = (r.stdout or "").strip()
            if s:
                return float(s)
        except Exception:
            pass
        return None

    def _wav_rms_envelope(self, wav_path: str, frame_sec: float = 0.05) -> List[float]:
        """
        Returns list of 0..1 mouth levels (rough envelope) for lipsync.
        WAV must be mono 16-bit. If not, caller should convert.
        """
        import wave
        try:
            with wave.open(wav_path, "rb") as wf:
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                if ch != 1 or sw != 2:
                    return []
                n = wf.getnframes()
                raw = wf.readframes(n)
        except Exception:
            return []

        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if x.size < 512:
            return []

        hop = max(1, int(sr * frame_sec))
        # normalize based on percentiles to avoid spikes
        env = []
        for i in range(0, len(x), hop):
            seg = x[i:i+hop]
            if seg.size < 64:
                break
            rms = float(np.sqrt(np.mean(seg * seg) + 1e-9))
            env.append(rms)

        if not env:
            return []

        a = np.array(env, dtype=np.float32)
        p90 = float(np.percentile(a, 90))
        p20 = float(np.percentile(a, 20))
        den = max(1e-6, (p90 - p20))
        out = []
        for v in a:
            lvl = (float(v) - p20) / den
            lvl = max(0.0, min(1.0, lvl))
            out.append(lvl)
        return out

    def _make_lipsync_envelope(self, audio_path: str) -> Tuple[List[float], float]:
        """
        Return (envelope_levels, frame_sec)
        For mp3/m4a: convert to temp wav (mono 16k) via ffmpeg.
        """
        frame_sec = 0.05
        p = audio_path.lower()

        # direct wav
        if p.endswith(".wav"):
            env = self._wav_rms_envelope(audio_path, frame_sec=frame_sec)
            return env, frame_sec

        # convert with ffmpeg to a temp wav for envelope
        tmpdir = tempfile.mkdtemp(prefix="lipsync_")
        wav_tmp = os.path.join(tmpdir, "tmp.wav")
        try:
            # ffmpeg: decode to mono 16k wav
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-vn", wav_tmp],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
            env = self._wav_rms_envelope(wav_tmp, frame_sec=frame_sec)
            return env, frame_sec
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    # ----------------------------
    # Audio play with mouth movement
    # ----------------------------
    def _play_audio_with_mouth(self, audio_path: str, emo: str = "suprise", loops: int = 1):
        """
        - Ensure face mode
        - Set emotion (suprise)
        - Play using robot_hat Music
        - While playing: send MOUTH values (envelope-based if possible)
        """
        self._playing_audio = True
        try:
            self.video.show_face()

            # set emotion before speaking
            if emo:
                self.face.emo(emo)

            dur = self._ffprobe_duration(audio_path)
            env, frame_sec = self._make_lipsync_envelope(audio_path)

            # start playback
            self.music.music_play(str(audio_path), loops=max(1, int(loops)))

            # If we can’t get duration, fallback to 6s
            total = dur if (dur is not None and dur > 0.1) else 6.0
            total *= max(1, int(loops))

            start = time.time()
            idx = 0
            while not self._stop:
                now = time.time()
                t = now - start
                if t >= total:
                    break

                # mouth level
                if env:
                    # map time -> env index
                    k = int(t / frame_sec)
                    if k < 0:
                        k = 0
                    if k >= len(env):
                        lvl = 0.0
                    else:
                        lvl = env[k]
                else:
                    # fallback: simple animated mouth
                    lvl = 0.15 + 0.45 * abs(math.sin(now * 12.0))

                self.face.mouth(lvl)
                time.sleep(0.02)

        except Exception as e:
            print("[AUDIO] play error:", e)
        finally:
            # stop mouth and return to neutral face
            try:
                self.face.mouth(0.0)
            except Exception:
                pass
            try:
                self.face.emo("what_is_it")
            except Exception:
                pass

            self._playing_audio = False
            self._block_until = time.time() + float(self.cfg.playback_cooldown_sec)

    # ----------------------------
    # Bark (sad + local wav)
    # ----------------------------
    def _bark(self):
        if not self._bark_path.exists():
            print(f"[WARN] bark file not found: {self._bark_path}")
            return

        # sad face while barking
        self.video.show_face()
        self.face.emo("sad")

        # bark sound (no need perfect lipsync, but we still can animate mouth a bit)
        self._play_audio_with_mouth(str(self._bark_path), emo="sad", loops=int(self.cfg.bark_times))

        # return to idle face
        self.face.emo("what_is_it")

    # ----------------------------
    # Noise calibration / gate
    # ----------------------------
    def _calibrate_noise_floor(self):
        secs = max(1.0, float(self.cfg.noise_calib_sec))
        n_chunks = int(math.ceil(secs / max(0.2, self.cfg.detect_chunk_sec)))
        samples = []

        print(f"[CALIB] measuring noise floor for ~{secs:.1f}s ... keep quiet")
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
            self._noise_rms = max(float(self.cfg.min_rms_floor), float(np.median(np.array(samples))))

        print(f"[CALIB] noise_rms≈{self._noise_rms:.0f} gate=+{self.cfg.gate_db_above_noise:.1f}dB")

    def _passes_gate(self, rms: float) -> bool:
        if self._noise_rms is None:
            return rms >= float(self.cfg.min_rms_floor)
        thr = self._noise_rms * (10.0 ** (float(self.cfg.gate_db_above_noise) / 20.0))
        return rms >= thr

    # ----------------------------
    # Audio record + features
    # ----------------------------
    def _record_wav(self, seconds: float) -> Optional[str]:
        if seconds <= 0:
            return None

        tmp = tempfile.NamedTemporaryFile(prefix="al_", suffix=".wav", delete=False)
        tmp.close()
        out = tmp.name

        cmd = [
            "arecord",
            "-D", self.cfg.mic_device,
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(self.cfg.sample_rate),
            "-d", str(max(1, int(math.ceil(seconds)))),
            out,
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

    def _extract_features(self, wav_path: str) -> Dict[str, float]:
        x = self._read_wav_pcm16(wav_path)
        if x is None or len(x) < 256:
            return {"rms": 0.0, "speech_score": 0.0, "flatness": 1.0,
                    "speech_ratio": 0.0, "high_ratio": 0.0, "zcr": 0.0, "peak_ratio": 0.0}

        rms = float(np.sqrt(np.mean(x * x) + 1e-9))
        peak = float(np.max(np.abs(x)) + 1e-9
                    )
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
            "zcr": float(zc),
            "peak_ratio": peak_ratio,
        }

    def _is_clap(self, feats: Dict[str, float]) -> bool:
        return (
            feats["peak_ratio"] >= float(self.cfg.clap_peak_ratio)
            and feats["high_ratio"] >= float(self.cfg.clap_high_ratio)
            and feats["zcr"] >= float(self.cfg.clap_zcr)
        )

    # ----------------------------
    # Camera
    # ----------------------------
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
            print("[MEM] write error:", e)

    # ----------------------------
    # Server
    # ----------------------------
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
            rr = requests.get(url, timeout=30)
            if rr.status_code != 200:
                return False
            with open(dst, "wb") as f:
                f.write(rr.content)
            return True
        except Exception:
            return False

    # ----------------------------
    # Handle server reply:
    # - play.youtube -> video service
    # - audio_url -> stop video -> face suprise + mouth move -> play audio
    # ----------------------------
    def _handle_server_reply(self, resp: Dict[str, Any]):
        transcript = (resp.get("transcript") or "").strip()
        label = (resp.get("label") or "unknown").strip()
        reply_text = (resp.get("reply_text") or "").strip()
        audio_url = resp.get("audio_url")
        play = resp.get("play")  # NEW

        print("[SERVER] label=", label)
        print("[USER ]", transcript)
        if reply_text:
            print("[BOT  ]", reply_text)
        print("[AUDIO]", audio_url)
        print("[PLAY ]", play)

        self._append_memory({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": transcript,
            "label": label,
            "reply_text": reply_text,
            "audio_url": audio_url,
            "play": play,
        })

        # clap label from server
        if label.lower() == "clap":
            self._bark()
            return

        # ---- YOUTUBE PLAY ----
        if isinstance(play, dict) and (play.get("type") == "youtube" or play.get("provider") == "youtube"):
            player_url = play.get("playerUrl") or play.get("player_url") or play.get("watchUrl") or play.get("watch_url")
            dur = play.get("duration_sec") or play.get("duration")  # optional
            if player_url:
                ok = self.video.play_youtube(player_url)
                print("[VIDEO] play ok?" , ok)

                # block mic to avoid self-triggering from music
                block_sec = float(dur) if isinstance(dur, (int, float)) and dur > 1 else float(self.cfg.youtube_default_block_sec)
                self._block_until = time.time() + block_sec
            return

        # ---- AUDIO (chat / tts) ----
        if not audio_url:
            # nothing to play: just ensure face mode
            self.video.show_face()
            self.face.emo("what_is_it")
            return

        # stop video if needed + show face
        self.video.stop_video()
        self.video.show_face()

        # download audio then play with surprise + mouth moving
        tmpdir = tempfile.mkdtemp(prefix="reply_")
        local = os.path.join(tmpdir, "reply.mp3")
        try:
            if not self._download(audio_url, local):
                print("[AUDIO] download failed:", audio_url)
                return
            self._play_audio_with_mouth(local, emo="suprise", loops=1)
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    # ----------------------------
    # Main loop
    # ----------------------------
    def run_forever(self):
        print("[ActiveListenerMain] start listening...")
        self._calibrate_noise_floor()

        # idle face
        self.video.show_face()
        self.face.emo("what_is_it")
        self.face.mouth(0.0)

        while not self._stop:
            # block while playing audio or while youtube is playing (block_until)
            if self._playing_audio or (time.time() < self._block_until):
                time.sleep(0.08)
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

                # clap -> bark
                if self._is_clap(feats):
                    print("[CLAP] -> bark")
                    self._bark()
                    continue

                # speech?
                if feats["speech_score"] >= self.cfg.speech_score_threshold:
                    print(f"[SPEECH] rms={rms:.0f} score={feats['speech_score']:.2f}")

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
                    # env noise -> bark (as your old logic)
                    print("[ENV] -> bark")
                    self._bark()

            finally:
                try:
                    os.remove(wav_path)
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
        playback_cooldown_sec=0.7,
        volume=80,
        face_udp_host="127.0.0.1",
        face_udp_port=39393,
        video_service_url="http://127.0.0.1:9900",
        youtube_default_block_sec=240.0,
    )

    al = ActiveListenerMain(cfg)

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
