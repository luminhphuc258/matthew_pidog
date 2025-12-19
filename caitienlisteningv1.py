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
from motion_controller import MotionController


@dataclass
class ListenerCfg:
    # audio
    mic_device: str = "default"
    sample_rate: int = 16000

    detect_chunk_sec: float = 0.6     # ngắn hơn để bắt clap tốt
    record_sec: float = 6.0

    # ===== auto noise gate =====
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

    # ✅ NEW: cooldown sau khi phát loa để khỏi tự kích hoạt lại
    playback_cooldown_sec: float = 0.7

    # ✅ NEW: volume (0-100)
    volume: int = 80


class ActiveListenerV2:
    def __init__(self, cfg: ListenerCfg):
        self.cfg = cfg
        self._playing = False
        self._stop = False

        self.music = Music()
        try:
            self.music.music_set_volume(int(self.cfg.volume))
        except Exception:
            pass

        self._mem_path = Path(self.cfg.memory_file)
        self._bark_path = Path(self.cfg.bark_wav)

        self._noise_rms: Optional[float] = None
        self._cooldown_until = 0.0

        # (optional) unlock speaker pin if needed
        try:
            os.system("pinctrl set 12 op dh")
            time.sleep(0.1)
        except Exception:
            pass

    def stop(self):
        self._stop = True

    # ---------- duration helper ----------
    def _get_audio_duration_sec(self, filepath: str) -> Optional[float]:
        p = str(filepath)
        # wav -> wave module
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

        # mp3/m4a -> ffprobe if exists
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
    def _music_play_blocking(self, filepath: str, times: int = 1):
        """
        Quan trọng:
        - Dùng loops=... đúng với robot_hat trên máy bạn
        - BLOCK bằng sleep theo duration để không bị cắt 1s
        - Trong lúc play: _playing=True => mic không record
        """
        self._playing = True
        try:
            dur = self._get_audio_duration_sec(filepath)
            loops = max(1, int(times))

            # play
            self.music.music_play(str(filepath), loops=loops)

            # block until done (best effort)
            if dur is not None:
                time.sleep(dur * loops + 0.15)
            else:
                # fallback nếu không đo được dur
                time.sleep(2.5 * loops)

        except Exception as e:
            print("[PLAY] error:", e)
        finally:
            self._playing = False
            self._cooldown_until = time.time() + float(self.cfg.playback_cooldown_sec)

    def _bark(self):
        if not self._bark_path.exists():
            print(f"[WARN] bark file not found: {self._bark_path}")
            return
        self._music_play_blocking(str(self._bark_path), times=int(self.cfg.bark_times))

    # ---------- MAIN ----------
    def run_forever(self):
        print("[ActiveListenerV2] start listening...")
        self._calibrate_noise_floor()

        while not self._stop:
            # ✅ Nếu đang phát loa / đang cooldown thì bỏ qua thu âm
            if self._playing or (time.time() < self._cooldown_until):
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
                    print(f"[CLAP] rms={rms:.0f} peak/rms={feats['peak_ratio']:.1f} hi={feats['high_ratio']:.3f} zcr={feats['zcr']:.3f} -> BARK")
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
            rr = requests.get(url, timeout=30)
            if rr.status_code != 200:
                return False
            with open(dst, "wb") as f:
                f.write(rr.content)
            return True
        except Exception:
            return False

    # ✅ UPDATED: play reply audio BLOCKING + mic off while playing
    def _handle_server_reply(self, resp: Dict[str, Any]):
        transcript = (resp.get("transcript") or "").strip()
        label = (resp.get("label") or "unknown").strip()
        reply_text = (resp.get("reply_text") or "").strip()
        audio_url = resp.get("audio_url")

        print("[SERVER] label=", label)
        print("[USER ]", transcript)
        if reply_text:
            print("[BOT  ]", reply_text)
        print("[AUDIO]", audio_url)

        self._append_memory({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": transcript,
            "label": label,
            "reply_text": reply_text,
            "audio_url": audio_url,
        })

        # nếu server trả label clap thì bark local
        if label.lower() == "clap":
            print(f"[CLAP->BARK] bark x{self.cfg.bark_times}")
            self._bark()
            return

        if not audio_url:
            return

        tmpdir = tempfile.mkdtemp(prefix="al2_play_")
        local = os.path.join(tmpdir, "reply.mp3")
        try:
            if not self._download(audio_url, local):
                print("[PLAY] download failed:", audio_url)
                return

            # ✅ phát BLOCKING để không cắt + mic off
            self._music_play_blocking(local, times=1)

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
        playback_cooldown_sec=0.7,
        volume=80,
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
