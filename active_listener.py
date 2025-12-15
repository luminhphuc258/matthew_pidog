#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import wave
import struct
import math
import tempfile
import subprocess
import threading
from typing import Optional, Dict, Callable
import shutil
import socket

import requests


class ActiveListener:
    """
    Active listening + auto-reply audio playback + lip-sync (UDP to face3d).

    Face3D service supports:
      - EMO <name>      (music/suprise/...)
      - MOUTH <0..1>    (mouth openness / audio energy)
    """

    def __init__(
        self,
        mic_device: str = "default",
        speaker_device: str = "default",
        sample_rate: int = 16000,
        detect_chunk_sec: int = 1,
        record_sec: int = 6,                  # ✅ ghi âm 6s
        threshold: int = 2700,
        cooldown_sec: float = 1.0,
        post_play_silence_sec: float = 0.25,  # đệm nhỏ sau khi phát

        nodejs_upload_url: str = "https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        on_result: Optional[Callable[[str, str, Optional[str]], None]] = None,
        debug: bool = False,

        # UDP to face3d
        face_host: str = "127.0.0.1",
        face_port: int = 39393,
        lipsync_fps: int = 50,
        lipsync_gain: float = 9000.0,         # RMS / gain -> 0..1 (tự chỉnh)
    ):
        self.mic_device = mic_device
        self.speaker_device = speaker_device
        self.sample_rate = int(sample_rate)
        self.detect_chunk_sec = int(detect_chunk_sec)
        self.record_sec = int(record_sec)
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

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # ✅ flag: đang phát âm thanh => không record/detect
        self._playing = threading.Event()

        # UDP socket (reuse)
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_addr = (self.face_host, self.face_port)

        # public state
        self.last_transcript: str = ""
        self.last_label: str = ""
        self.last_audio_url: Optional[str] = None
        self.last_ts: float = 0.0

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
        """Send one UDP command line to face3d service."""
        try:
            self._udp.sendto(msg.encode("utf-8"), self._udp_addr)
        except Exception:
            pass

    def _record_wav(self, filename: str, seconds: int) -> bool:
        cmd = [
            "arecord",
            "-D", self.mic_device,
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", "1",
            "-d", str(int(seconds)),
            "-q",
            filename,
        ]
        return subprocess.run(cmd, check=False).returncode == 0

    def _get_max_amplitude(self, filename: str) -> int:
        try:
            with wave.open(filename, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                if not raw:
                    return 0
            samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
            return max(abs(s) for s in samples) if samples else 0
        except Exception:
            return 0

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

    # ---------------- lip-sync helpers ----------------

    def _ensure_wav(self, filepath: str) -> (Optional[str], bool):
        """
        Ensure we have a WAV file for lip-sync RMS reading.
        Returns (wav_path, created_wav).
        """
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
        """
        Stream mouth levels while _playing is set.
        Sends: MOUTH <0..1> at lipsync_fps
        """
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

                # if stereo, pick first channel
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

    def _play_audio_with_lipsync(self, filepath: str):
        """
        Playback audio + lipsync.
        - _playing ON before playback -> loop pauses listening.
        - Face EMO music during playback.
        """
        self._playing.set()
        wav_path = None
        created_wav = False

        try:
            # ✅ Face when speaking
            self.send_face_cmd("EMO music")

            wav_path, created_wav = self._ensure_wav(filepath)

            # lipsync thread (if wav exists)
            lip_thread = None
            if wav_path and os.path.exists(wav_path):
                lip_thread = threading.Thread(target=self._rms_stream_wav, args=(wav_path,), daemon=True)
                lip_thread.start()

            # play blocking
            if wav_path and os.path.exists(wav_path):
                subprocess.run(["aplay", "-D", self.speaker_device, "-q", wav_path], check=False)
            else:
                # fallback: mpg123 if mp3
                if filepath.lower().endswith(".mp3") and shutil.which("mpg123"):
                    subprocess.run(["mpg123", "-q", "-a", self.speaker_device, filepath], check=False)
                else:
                    subprocess.run(["aplay", "-D", self.speaker_device, "-q", filepath], check=False)

            if lip_thread:
                lip_thread.join(timeout=0.6)

        finally:
            self._playing.clear()

            if self.post_play_silence_sec > 0:
                time.sleep(self.post_play_silence_sec)

            # cleanup wav created from mp3
            try:
                if created_wav and wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass

            # ensure mouth closed
            self.send_face_cmd("MOUTH 0.0")

    # ---------------- main loop ----------------

    def _loop(self):
        self._log("start", "mic=", self.mic_device, "spk=", self.speaker_device)

        while not self._stop.is_set():
            if self._playing.is_set():
                time.sleep(0.05)
                continue

            # 1) detect chunk
            fd, detect_file = tempfile.mkstemp(suffix=".wav", prefix="det_")
            os.close(fd)

            ok = self._record_wav(detect_file, self.detect_chunk_sec)
            level = self._get_max_amplitude(detect_file) if ok else 0
            try:
                os.unlink(detect_file)
            except Exception:
                pass

            if self.debug:
                self._log("level=", level)

            if level < self.threshold:
                time.sleep(0.03)
                continue

            # cooldown chống spam
            now = time.time()
            if now - self.last_ts < self.cooldown_sec:
                time.sleep(0.05)
                continue

            if self._playing.is_set():
                time.sleep(0.05)
                continue

            # ✅ Face when listening/recording
            self.send_face_cmd("EMO suprise")
            self.send_face_cmd("MOUTH 0.0")

            # 2) record full (6s)
            fd, record_file = tempfile.mkstemp(suffix=".wav", prefix="voice_")
            os.close(fd)

            self._log("TRIGGER -> recording", self.record_sec)
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

            transcript = resp.get("transcript", "") or ""
            label = resp.get("label", "") or ""
            audio_url = resp.get("audio_url", None)

            self.last_transcript = transcript
            self.last_label = label
            self.last_audio_url = audio_url
            self.last_ts = time.time()

            if self.on_result:
                try:
                    self.on_result(transcript, label, audio_url)
                except Exception as e:
                    self._log("on_result error:", e)

            # 4) download + play
            if audio_url and not self._stop.is_set():
                reply = self._download(audio_url)
                if reply:
                    try:
                        self._play_audio_with_lipsync(reply)
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
