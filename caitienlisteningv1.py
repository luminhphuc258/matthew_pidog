#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================
# FIX OPENCV / GSTREAMER
# ==========================
import os
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")  # disable gstreamer backend priority
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")              # reduce OpenCV logs

import time
import json
import math
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import requests

try:
    import cv2
except Exception:
    cv2 = None

from robot_hat import Music

# ==== BOOT via MotionController ====
from motion_controller import MotionController


@dataclass
class ListenerCfg:
    # audio
    mic_device: str = "default"
    sample_rate: int = 16000
    detect_chunk_sec: float = 0.9
    record_sec: float = 4.0
    min_rms: float = 900.0
    speech_score_threshold: float = 0.55

    # server
    server_url: str = "https://embeddedprogramming-healtheworldserver.up.railway.app/pi_upload_audio_v2"
    timeout_sec: float = 30.0

    # camera (Brio)
    cam_dev: str = "/dev/video0"
    cam_w: int = 640
    cam_h: int = 480
    jpeg_quality: int = 80
    cam_warmup_frames: int = 6
    cam_backend: str = "v4l2"  # force v4l2

    # memory
    memory_file: str = "robot_memory.jsonl"
    memory_max_items_send: int = 12

    # bark
    bark_wav: str = "tiengsua.wav"
    bark_times: int = 2


class ActiveListenerV2:
    """
    Active listening v2:
    - record chunk ngắn -> classify env vs speech (heuristic)
    - env => bark (Music.music_play wav, loop=bark_times)
    - speech => record full -> send server (audio + optional image + memory)
    - long-term memory saved to jsonl and sent to server each request
    """

    def __init__(self, cfg: ListenerCfg):
        self.cfg = cfg
        self._playing = False
        self._stop = False

        self.music = Music()
        self._mem_path = Path(self.cfg.memory_file)
        self._bark_path = Path(self.cfg.bark_wav)

    def stop(self):
        self._stop = True

    def run_forever(self):
        print("[ActiveListenerV2] start listening...")
        while not self._stop:
            if self._playing:
                time.sleep(0.05)
                continue

            wav_path = self._record_wav(seconds=self.cfg.detect_chunk_sec)
            if not wav_path:
                time.sleep(0.1)
                continue

            try:
                rms, score, dbg = self._classify_chunk(wav_path)
                if rms < self.cfg.min_rms:
                    continue

                if score < self.cfg.speech_score_threshold:
                    print(f"[ENV] rms={rms:.0f} score={score:.2f} dbg={dbg} -> BARK x{self.cfg.bark_times}")
                    self._bark()
                    continue

                print(f"[SPEECH] rms={rms:.0f} score={score:.2f} dbg={dbg} -> record full")
                full_wav = self._record_wav(seconds=self.cfg.record_sec)
                if not full_wav:
                    continue

                # capture camera frame (Brio safe)
                image_bytes = self._capture_jpeg_frame()

                resp = self._send_to_server(full_wav, image_bytes=image_bytes)
                if not resp:
                    continue

                self._handle_server_reply(resp)

                try:
                    os.remove(full_wav)
                except Exception:
                    pass

            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    # -------------------- audio record --------------------
    def _record_wav(self, seconds: float) -> Optional[str]:
        """Record WAV (S16_LE mono 16k) using arecord."""
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
                nchan = wf.getnchannels()
                sampw = wf.getsampwidth()
                if nchan != 1 or sampw != 2:
                    return None
                raw = wf.readframes(wf.getnframes())
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        except Exception:
            return None

    # -------------------- classification --------------------
    def _classify_chunk(self, wav_path: str) -> Tuple[float, float, Dict[str, float]]:
        x = self._read_wav_pcm16(wav_path)
        if x is None or len(x) < 256:
            return 0.0, 0.0, {}

        rms = float(np.sqrt(np.mean(x * x) + 1e-9))
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
        score += (1.0 - min(1.0, flatness * 3.0)) * 0.45
        score += min(1.0, speech_ratio * 3.0) * 0.45
        score += (1.0 - min(1.0, high_ratio * 6.0)) * 0.10
        score = float(max(0.0, min(1.0, score)))

        dbg = {
            "flatness": float(flatness),
            "speech_ratio": float(speech_ratio),
            "high_ratio": float(high_ratio),
            "low_ratio": float(low_ratio),
            "zcr": float(zc),
        }
        return rms, score, dbg

    # -------------------- bark --------------------
    def _bark(self):
        if not self._bark_path.exists():
            print(f"[WARN] bark file not found: {self._bark_path}")
            return
        self._playing = True
        try:
            self.music.music_play(str(self._bark_path), loop=int(self.cfg.bark_times))
        except Exception as e:
            print("[BARK] error:", e)
        finally:
            self._playing = False

    # -------------------- camera (Brio safe) --------------------
    def _capture_jpeg_frame(self) -> Optional[bytes]:
        if cv2 is None:
            return None

        cap = None
        try:
            backend = cv2.CAP_V4L2 if self.cfg.cam_backend.lower() == "v4l2" else 0
            cap = cv2.VideoCapture(self.cfg.cam_dev, backend)
            if not cap.isOpened():
                print("[CAM] cannot open:", self.cfg.cam_dev)
                return None

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_h)

            # Brio often supports MJPG; try to set it (ignore if fails)
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
                print("[CAM] read failed")
                return None

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)]
            ok2, buf = cv2.imencode(".jpg", frame, encode_param)
            if not ok2:
                print("[CAM] encode failed")
                return None
            return buf.tobytes()

        except Exception as e:
            print("[CAM] error:", e)
            return None
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    # -------------------- memory --------------------
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
        return items[-self.cfg.memory_max_items_send :]

    def _append_memory(self, entry: Dict[str, Any]):
        try:
            with self._mem_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print("[MEM] write error:", e)

    # -------------------- server --------------------
    def _send_to_server(self, wav_path: str, image_bytes: Optional[bytes]) -> Optional[Dict[str, Any]]:
        mem = self._load_recent_memory()
        meta = {
            "ts": time.time(),
            "client": "pidog",
            "memory": mem,
        }

        files = {
            "audio": ("audio.wav", open(wav_path, "rb"), "audio/wav"),
        }
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

    def _play_audio_file(self, filepath: str):
        self._playing = True
        try:
            self.music.music_play(filepath, loop=1)
        except Exception as e:
            print("[PLAY] error:", e)
        finally:
            self._playing = False

    def _handle_server_reply(self, resp: Dict[str, Any]):
        transcript = (resp.get("transcript") or "").strip()
        label = resp.get("label") or "unknown"
        reply_text = (resp.get("reply_text") or "").strip()
        audio_url = resp.get("audio_url")

        print("[SERVER] label=", label)
        print("[USER ]", transcript)
        print("[BOT  ]", reply_text)
        print("[AUDIO]", audio_url)

        self._append_memory({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": transcript,
            "label": label,
            "reply_text": reply_text,
            "audio_url": audio_url,
        })

        if not audio_url:
            return

        tmpdir = tempfile.mkdtemp(prefix="al2_play_")
        local = os.path.join(tmpdir, "reply.mp3")
        try:
            if self._download(audio_url, local):
                self._play_audio_file(local)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================
# MAIN: BOOT using MotionController (unlock speaker)
# ==========================================================
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
        cam_dev="/dev/video0",   # Brio thường là /dev/video0 hoặc /dev/video2 tuỳ máy
        cam_w=640,
        cam_h=480,
        cam_backend="v4l2",
        cam_warmup_frames=6,
        memory_file="robot_memory.jsonl",
    )

    al = ActiveListenerV2(cfg)

    try:
        al.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        al.stop()
        # MotionController có/không có close tuỳ project bạn
        try:
            mc.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
