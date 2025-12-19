#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import time
import wave
import struct
import math
import tempfile
import subprocess
import threading
from typing import Optional, Dict, Callable, Tuple
import shutil
import socket

import requests
import numpy as np

from robot_hat import Music


class ActiveListener:
    """
    Drop-in replacement for your old ActiveListener:
    - same constructor signature (keeps other files working)
    - same public methods: start/stop/join/is_playing
    - keeps UDP Face3D: EMO <name>, MOUTH <0..1>

    NEW BEHAVIOR:
    - clap detection -> bark locally by robot_hat.Music (plays tiengsua.wav fully)
    - while playing any audio -> mic will not record (avoid self-trigger)
    - speech -> record full -> upload -> download reply -> play with lipsync
    """

    def __init__(
        self,
        mic_device: str = "default",
        speaker_device: str = "default",   # kept for compatibility (not used by Music)
        sample_rate: int = 16000,
        detect_chunk_sec: int = 1,
        record_sec: int = 6,
        threshold: int = 2700,            # kept for compatibility (old max amplitude)
        cooldown_sec: float = 1.0,
        post_play_silence_sec: float = 0.25,

        nodejs_upload_url: str = "https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        on_result: Optional[Callable[[str, str, Optional[str]], None]] = None,
        debug: bool = False,

        # UDP to face3d
        face_host: str = "127.0.0.1",
        face_port: int = 39393,
        lipsync_fps: int = 50,
        lipsync_gain: float = 9000.0,

        # ===== NEW / SAFE DEFAULTS (won't break older calls) =====
        bark_wav: str = "tiengsua.wav",
        bark_times: int = 2,
        noise_calib_sec: float = 2.0,
        gate_db_above_noise: float = 10.0,
        min_rms_floor: float = 700.0,
        speech_score_threshold: float = 0.62,
        clap_peak_ratio: float = 5.0,
        clap_high_ratio: float = 0.12,
        clap_zcr: float = 0.10,
        playback_cooldown_sec: float = 0.7,
        volume: int = 80,
    ):
        # keep old fields
        self.mic_device = mic_device
        self.speaker_device = speaker_device
        self.sample_rate = int(sample_rate)
        self.detect_chunk_sec = float(detect_chunk_sec)
        self.record_sec = float(record_sec)
        self.threshold = int(threshold)
        self.cooldown_sec = float(cooldown_sec)
        self.post_play_silence_sec = float(post_play_silence_sec)

        self.nodejs_upload_url = nodejs_upload_url
        self.on_result = on_result
        self.debug = bool(debug)

        self.face_host = face_host
        self.face_port = int(face_port)
        self.lipsync_fps = int(lipsync_fps)
        self.lipsync_gain = float(lipsync_gain)

        # NEW tuning
        self.bark_wav = str(bark_wav)
        self.bark_times = int(bark_times)
        self.noise_calib_sec = float(noise_calib_sec)
        self.gate_db_above_noise = float(gate_db_above_noise)
        self.min_rms_floor = float(min_rms_floor)
        self.speech_score_threshold = float(speech_score_threshold)

        self.clap_peak_ratio = float(clap_peak_ratio)
        self.clap_high_ratio = float(clap_high_ratio)
        self.clap_zcr = float(clap_zcr)

        self.playback_cooldown_sec = float(playback_cooldown_sec)
        self.volume = int(volume)

        # threading state
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # ✅ while playing audio -> no record
        self._playing = threading.Event()
        self._cooldown_until = 0.0

        # UDP socket
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_addr = (self.face_host, self.face_port)

        # public state
        self.last_transcript: str = ""
        self.last_label: str = ""
        self.last_audio_url: Optional[str] = None
        self.last_ts: float = 0.0

        # noise floor
        self._noise_rms: Optional[float] = None

        # robot_hat Music
        self.music = Music()
        try:
            self.music.music_set_volume(self.volume)
        except Exception:
            pass

        # unlock speaker pin (optional)
        try:
            os.system("pinctrl set 12 op dh")
            time.sleep(0.1)
        except Exception:
            pass

    # ---------------- public ----------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ActiveListener", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout: Optional[float] = 2.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    def is_playing(self) -> bool:
        return self._playing.is_set()

    # ---------------- utils ----------------
    def _log(self, *a):
        if self.debug:
            print("[LISTENER]", *a, flush=True)

    def send_face_cmd(self, msg: str):
        try:
            self._udp.sendto(msg.encode("utf-8"), self._udp_addr)
        except Exception:
            pass

    # ---------------- audio record ----------------
    def _record_wav(self, filename: str, seconds: float) -> bool:
        cmd = [
            "arecord",
            "-D", self.mic_device,
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", "1",
            "-d", str(max(1, int(math.ceil(seconds)))),
            "-q",
            filename,
        ]
        return subprocess.run(cmd, check=False).returncode == 0

    def _read_wav_pcm16(self, filename: str) -> Optional[np.ndarray]:
        try:
            with wave.open(filename, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    return None
                raw = wf.readframes(wf.getnframes())
            if not raw:
                return None
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        except Exception:
            return None

    # ---------------- feature extract ----------------
    def _extract_features(self, wav_path: str) -> Dict[str, float]:
        x = self._read_wav_pcm16(wav_path)
        if x is None or len(x) < 256:
            return {
                "rms": 0.0, "speech_score": 0.0,
                "flatness": 1.0, "speech_ratio": 0.0,
                "high_ratio": 0.0, "low_ratio": 0.0,
                "zcr": 0.0, "peak_ratio": 0.0
            }

        rms = float(np.sqrt(np.mean(x * x) + 1e-9))
        peak = float(np.max(np.abs(x)) + 1e-9)
        peak_ratio = float(peak / (rms + 1e-9))

        y = x / (np.max(np.abs(x)) + 1e-9)
        zcr = float(np.mean(y[1:] * y[:-1] < 0.0))

        sr = self.sample_rate
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
            "zcr": float(zcr),
            "peak_ratio": float(peak_ratio),
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

    # ---------------- gate + clap ----------------
    def _calibrate_noise_floor(self):
        secs = max(1.0, self.noise_calib_sec)
        n_chunks = int(math.ceil(secs / max(0.2, self.detect_chunk_sec)))
        samples = []

        print(f"[CALIB] measuring noise floor for ~{secs:.1f}s ... keep quiet if possible")
        for _ in range(max(2, n_chunks)):
            if self._stop.is_set():
                break
            fd, p = tempfile.mkstemp(suffix=".wav", prefix="cal_")
            os.close(fd)
            try:
                if not self._record_wav(p, self.detect_chunk_sec):
                    continue
                x = self._read_wav_pcm16(p)
                if x is None or len(x) < 256:
                    continue
                rms = float(np.sqrt(np.mean(x * x) + 1e-9))
                samples.append(rms)
            finally:
                try:
                    os.unlink(p)
                except Exception:
                    pass

        if not samples:
            self._noise_rms = float(self.min_rms_floor)
        else:
            med = float(np.median(np.array(samples)))
            self._noise_rms = max(float(self.min_rms_floor), med)

        print(f"[CALIB] noise_rms≈{self._noise_rms:.0f}  gate=+{self.gate_db_above_noise:.1f}dB")

    def _passes_gate(self, rms: float) -> bool:
        if self._noise_rms is None:
            return rms >= float(self.min_rms_floor)
        thr = self._noise_rms * (10.0 ** (self.gate_db_above_noise / 20.0))
        return rms >= thr

    def _is_clap(self, feats: Dict[str, float]) -> bool:
        return (
            feats["peak_ratio"] >= self.clap_peak_ratio
            and feats["high_ratio"] >= self.clap_high_ratio
            and feats["zcr"] >= self.clap_zcr
        )

    # ---------------- server io ----------------
    def _upload(self, wav_path: str) -> Optional[Dict]:
        try:
            with open(wav_path, "rb") as f:
                files = {"audio": ("voice.wav", f, "audio/wav")}
                resp = requests.post(self.nodejs_upload_url, files=files, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self._log("upload error:", e)
            return None

    def _download(self, url: str) -> Optional[str]:
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            fd, path = tempfile.mkstemp(suffix=".mp3", prefix="reply_")
            os.close(fd)
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return path
        except Exception as e:
            self._log("download error:", e)
            return None

    # ---------------- lipsync helpers ----------------
    def _ensure_wav(self, filepath: str) -> Tuple[Optional[str], bool]:
        if filepath.lower().endswith(".wav") and os.path.exists(filepath):
            return filepath, False

        if filepath.lower().endswith(".mp3") and os.path.exists(filepath):
            if not shutil.which("ffmpeg"):
                return None, False
            wav_path = filepath[:-4] + ".wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", filepath, "-ac", "1", "-ar", str(self.sample_rate), wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if os.path.exists(wav_path):
                return wav_path, True

        return None, False

    def _rms_stream_wav(self, wav_path: str):
        fps = max(10, self.lipsync_fps)
        dt = 1.0 / fps

        try:
            wf = wave.open(wav_path, "rb")
            rate = wf.getframerate() or self.sample_rate
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            if sw != 2:
                wf.close()
                return

            frames_per = int(rate * dt)
            if frames_per <= 0:
                frames_per = 320

            while self._playing.is_set() and not self._stop.is_set():
                frames = wf.readframes(frames_per)
                if not frames:
                    break

                n = len(frames) // 2
                if n <= 0:
                    time.sleep(dt)
                    continue

                samples = struct.unpack("<" + "h" * n, frames)
                if ch > 1:
                    samples = samples[::ch]

                ss = 0.0
                for s in samples:
                    ss += float(s) * float(s)
                rms = math.sqrt(ss / max(1, len(samples)))

                lvl = rms / self.lipsync_gain
                if lvl < 0.0:
                    lvl = 0.0
                elif lvl > 1.0:
                    lvl = 1.0

                self.send_face_cmd(f"MOUTH {lvl:.3f}")
                time.sleep(dt)

            wf.close()

        except Exception:
            pass
        finally:
            self.send_face_cmd("MOUTH 0.0")

    # ---------------- music playback (robot_hat) ----------------
    def _get_audio_duration_sec(self, filepath: str) -> Optional[float]:
        p = str(filepath)
        if p.lower().endswith(".wav"):
            try:
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

    def _music_play_blocking(self, filepath: str, loops: int = 1):
        self._playing.set()
        try:
            self.send_face_cmd("EMO music")

            dur = self._get_audio_duration_sec(filepath)
            loops = max(1, int(loops))

            # ✅ IMPORTANT: use loops= (your robot_hat expects loops)
            self.music.music_play(str(filepath), loops=loops)

            if dur is not None:
                time.sleep(dur * loops + 0.15)
            else:
                time.sleep(2.5 * loops)

        except Exception as e:
            self._log("music_play error:", e)
        finally:
            self._playing.clear()
            self._cooldown_until = time.time() + float(self.playback_cooldown_sec)
            if self.post_play_silence_sec > 0:
                time.sleep(self.post_play_silence_sec)
            self.send_face_cmd("MOUTH 0.0")

    def _bark(self):
        if not os.path.exists(self.bark_wav):
            print(f"[WARN] bark file not found: {self.bark_wav}")
            return
        self._music_play_blocking(self.bark_wav, loops=self.bark_times)

    def _play_reply_with_lipsync(self, mp3_path: str):
        """
        Reply audio from server is mp3.
        We:
        - convert to wav for lipsync RMS
        - play with Music (robot_hat)
        - lipsync thread uses wav
        """
        self._playing.set()
        wav_path = None
        created_wav = False
        lip_thread = None

        try:
            self.send_face_cmd("EMO music")
            wav_path, created_wav = self._ensure_wav(mp3_path)

            if wav_path and os.path.exists(wav_path):
                lip_thread = threading.Thread(target=self._rms_stream_wav, args=(wav_path,), daemon=True)
                lip_thread.start()

            # play mp3 by Music (blocking with duration)
            dur = self._get_audio_duration_sec(mp3_path)
            self.music.music_play(mp3_path, loops=1)
            if dur is not None:
                time.sleep(dur + 0.15)
            else:
                time.sleep(3.0)

            if lip_thread:
                lip_thread.join(timeout=0.6)

        finally:
            self._playing.clear()
            self._cooldown_until = time.time() + float(self.playback_cooldown_sec)

            if self.post_play_silence_sec > 0:
                time.sleep(self.post_play_silence_sec)

            try:
                if created_wav and wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass

            self.send_face_cmd("MOUTH 0.0")

    # ---------------- main loop ----------------
    def _loop(self):
        self._log("start", "mic=", self.mic_device, "spk=", self.speaker_device)

        # learn noise floor once at start
        self._calibrate_noise_floor()

        while not self._stop.is_set():
            # if playing or cooldown -> don't record
            if self._playing.is_set() or (time.time() < self._cooldown_until):
                time.sleep(0.05)
                continue

            # 1) detect chunk
            fd, detect_file = tempfile.mkstemp(suffix=".wav", prefix="det_")
            os.close(fd)

            ok = self._record_wav(detect_file, self.detect_chunk_sec)
            feats = None
            if ok:
                feats = self._extract_features(detect_file)

            try:
                os.unlink(detect_file)
            except Exception:
                pass

            if not feats:
                time.sleep(0.03)
                continue

            rms = feats["rms"]
            if not self._passes_gate(rms):
                time.sleep(0.02)
                continue

            # optional: keep old threshold behavior as extra safety (max amplitude)
            # (not required anymore, but doesn't hurt)
            if self.threshold > 0:
                # approximate max amplitude from rms+peak_ratio
                approx_peak = rms * feats.get("peak_ratio", 0.0)
                if approx_peak < self.threshold:
                    time.sleep(0.02)
                    continue

            # cooldown anti-spam
            now = time.time()
            if now - self.last_ts < self.cooldown_sec:
                time.sleep(0.05)
                continue

            # clap first
            if self._is_clap(feats):
                print(f"[CLAP] rms={rms:.0f} dbg={self._dbg_dict(feats)} -> BARK")
                self.last_ts = time.time()
                self._bark()
                continue

            # speech check
            if feats["speech_score"] < self.speech_score_threshold:
                # env noise => ignore (or bark if you want). For safety: ignore.
                self._log("env noise", self._dbg_dict(feats))
                time.sleep(0.03)
                continue

            # 2) record full
            if self._playing.is_set() or self._stop.is_set():
                time.sleep(0.05)
                continue

            self.send_face_cmd("EMO suprise")
            self.send_face_cmd("MOUTH 0.0")

            fd, record_file = tempfile.mkstemp(suffix=".wav", prefix="voice_")
            os.close(fd)

            self._log("SPEECH -> recording", self.record_sec, "dbg=", self._dbg_dict(feats))
            self._record_wav(record_file, self.record_sec)

            # 3) upload
            resp = self._upload(record_file)
            try:
                os.unlink(record_file)
            except Exception:
                pass

            if not resp:
                time.sleep(0.1)
                continue

            transcript = (resp.get("transcript") or "").strip()
            label = (resp.get("label") or "").strip()
            audio_url = resp.get("audio_url", None)

            print("[SERVER] label=", label)
            print("[USER ]", transcript)
            print("[AUDIO]", audio_url)

            self.last_transcript = transcript
            self.last_label = label
            self.last_audio_url = audio_url
            self.last_ts = time.time()

            if self.on_result:
                try:
                    self.on_result(transcript, label, audio_url)
                except Exception as e:
                    self._log("on_result error:", e)

            # 3.5) if server says clap -> bark locally (optional contract)
            if (label or "").lower() == "clap":
                print("[CLAP label from server] -> BARK")
                self._bark()
                continue

            # 4) download + play (mic off while playing)
            if audio_url and not self._stop.is_set():
                reply = self._download(audio_url)
                if reply:
                    try:
                        self._play_reply_with_lipsync(reply)
                    finally:
                        try:
                            os.unlink(reply)
                        except Exception:
                            pass

            time.sleep(0.05)

        self._log("stop")
        try:
            self.send_face_cmd("MOUTH 0.0")
        except Exception:
            pass
