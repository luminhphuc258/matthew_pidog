#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, wave, struct, tempfile, subprocess
import threading
import requests
from typing import Optional, Dict


class ActiveListener:
    def __init__(
        self,
        mic_device="plughw:3,0",
        speaker_device="plughw:1,0",
        sample_rate=16000,
        detect_chunk_sec=1,
        record_sec=4,
        threshold=2500,
        nodejs_upload_url="https://embeddedprogramming-healtheworldserver.up.railway.app/upload_audio",
    ):
        self.mic_device = mic_device
        self.speaker_device = speaker_device
        self.sample_rate = int(sample_rate)
        self.detect_chunk_sec = int(detect_chunk_sec)
        self.record_sec = int(record_sec)
        self.threshold = int(threshold)
        self.nodejs_upload_url = nodejs_upload_url

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.last_transcript = ""
        self.last_label = ""
        self.last_audio_url = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _record_wav(self, filename, seconds):
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
        subprocess.run(cmd, check=False)

    def _get_max_amplitude(self, filename) -> int:
        try:
            with wave.open(filename, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                if not raw:
                    return 0
            samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
            return max(abs(s) for s in samples)
        except Exception:
            return 0

    def _upload(self, filepath) -> Optional[Dict]:
        try:
            with open(filepath, "rb") as f:
                files = {"audio": ("voice.wav", f, "audio/wav")}
                resp = requests.post(self.nodejs_upload_url, files=files, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def _download(self, url) -> Optional[str]:
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
        except Exception:
            return None

    def _convert_mp3_to_wav(self, mp3_path):
        wav_path = mp3_path.replace(".mp3", ".wav")
        cmd = ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", str(self.sample_rate), wav_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path

    def _play(self, filepath):
        if filepath.endswith(".mp3"):
            filepath = self._convert_mp3_to_wav(filepath)
        cmd = ["aplay", "-D", self.speaker_device, "-q", filepath]
        subprocess.run(cmd, check=False)

    def _loop(self):
        while not self._stop.is_set():
            # 1) detect
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            detect_file = tmp.name
            tmp.close()

            self._record_wav(detect_file, self.detect_chunk_sec)
            level = self._get_max_amplitude(detect_file)
            try:
                os.unlink(detect_file)
            except Exception:
                pass

            if level < self.threshold:
                time.sleep(0.05)
                continue

            # 2) record full sentence
            fd, record_file = tempfile.mkstemp(suffix=".wav", prefix="voice_")
            os.close(fd)
            self._record_wav(record_file, self.record_sec)

            # 3) upload
            resp = self._upload(record_file)
            try:
                os.unlink(record_file)
            except Exception:
                pass
            if not resp:
                time.sleep(0.2)
                continue

            self.last_transcript = resp.get("transcript", "")
            self.last_label = resp.get("label", "")
            self.last_audio_url = resp.get("audio_url")

            # 4) download + play
            if self.last_audio_url:
                reply = self._download(self.last_audio_url)
                if reply:
                    self._play(reply)
                    try:
                        os.unlink(reply)
                    except Exception:
                        pass

            time.sleep(0.2)
