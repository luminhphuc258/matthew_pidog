#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import wave
import struct
import tempfile
import subprocess
import threading
from typing import Optional, Dict, Callable
import shutil

import requests


class ActiveListener:
    def __init__(
        self,
        mic_device: str = "default",
        speaker_device: str = "default",
        sample_rate: int = 16000,
        detect_chunk_sec: int = 1,
        record_sec: int = 6,                 # ✅ ghi âm 6s
        threshold: int = 2700,
        cooldown_sec: float = 1.0,
        post_play_silence_sec: float = 0.3,  # ✅ đệm nhỏ (tuỳ bạn)
        nodejs_upload_url: str = "https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
        on_result: Optional[Callable[[str, str, Optional[str]], None]] = None,
        debug: bool = False,
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

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # ✅ flag: đang phát âm thanh => không record
        self._playing = threading.Event()

        self.last_transcript: str = ""
        self.last_label: str = ""
        self.last_audio_url: Optional[str] = None
        self.last_ts: float = 0.0

    # ------------- public -------------

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

    # ------------- utils -------------

    def _log(self, *a):
        if self.debug:
            print("[LISTENER]", *a, flush=True)

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

    # ===== NEW: estimate duration of returned audio =====
    def _audio_duration_sec(self, filepath: str) -> Optional[float]:
        """
        Ưu tiên ffprobe để lấy duration chính xác.
        Fallback: nếu là wav thì đọc header.
        """
        # 1) ffprobe (best)
        try:
            if shutil.which("ffprobe"):
                p = subprocess.run(
                    ["ffprobe", "-v", "error",
                     "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1",
                     filepath],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                s = (p.stdout or "").strip()
                if s:
                    return float(s)
        except Exception:
            pass

        # 2) WAV header fallback
        try:
            if filepath.lower().endswith(".wav"):
                with wave.open(filepath, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or self.sample_rate
                    if rate > 0:
                        return float(frames) / float(rate)
        except Exception:
            pass

        return None

    def _play_audio(self, filepath: str):
        """
        ✅ Mic sẽ tắt đúng bằng thời lượng file audio server trả về:
        - set _playing trước khi play
        - play BLOCKING
        - sau khi play xong, sleep thêm (duration + post_play_silence_sec)
        """
        # đo duration ngay từ đầu (mp3/wav)
        dur = self._audio_duration_sec(filepath)

        self._playing.set()
        t0 = time.time()
        try:
            # mp3 -> mpg123 (blocking)
            if filepath.endswith(".mp3") and shutil.which("mpg123"):
                subprocess.run(
                    ["mpg123", "-q", "-a", self.speaker_device, filepath],
                    check=False
                )
            else:
                # fallback: mp3 -> wav -> aplay (blocking)
                if filepath.endswith(".mp3"):
                    wav_path = filepath[:-4] + ".wav"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", filepath, "-ac", "1", "-ar", str(self.sample_rate), wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    filepath = wav_path
                    # nếu lúc đầu chưa có duration, đo lại wav
                    if dur is None:
                        dur = self._audio_duration_sec(filepath)

                subprocess.run(["aplay", "-D", self.speaker_device, "-q", filepath], check=False)

        finally:
            # play elapsed thực tế
            elapsed = max(0.0, time.time() - t0)

            # nếu không đo được duration, fallback dùng elapsed (thực tế)
            target = dur if (dur is not None and dur > 0) else elapsed

            # ✅ đảm bảo mic tắt đúng bằng độ dài audio (không âm)
            extra = max(0.0, target - elapsed)

            # tắt mic thêm đúng phần còn thiếu + đệm
            if extra > 0:
                time.sleep(extra)
            if self.post_play_silence_sec > 0:
                time.sleep(self.post_play_silence_sec)

            self._playing.clear()

            # cleanup wav temp nếu có
            if filepath.endswith(".wav") and "reply_" in os.path.basename(filepath):
                try:
                    os.unlink(filepath)
                except Exception:
                    pass

    # ------------- main loop -------------

    def _loop(self):
        self._log("start", "mic=", self.mic_device, "spk=", self.speaker_device)

        while not self._stop.is_set():
            # ✅ nếu đang phát âm thanh => không ghi âm/detect
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

            # 4) download + play (mic off depends on audio duration)
            if audio_url and not self._stop.is_set():
                reply = self._download(audio_url)
                if reply:
                    self._play_audio(reply)
                    try:
                        os.unlink(reply)
                    except Exception:
                        pass

            time.sleep(0.05)

        self._log("stop")
