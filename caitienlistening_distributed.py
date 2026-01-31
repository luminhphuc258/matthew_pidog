#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock

import requests
from vosk import Model, KaldiRecognizer

try:
    from robot_hat import Music
except Exception:
    Music = None
try:
    from motion_controller import MotionController
except Exception:
    MotionController = None


def _require_arecord():
    if shutil.which("arecord") is None:
        print("ERROR: 'arecord' not found. Install alsa-utils.", flush=True)
        print("Example: sudo apt-get install -y alsa-utils", flush=True)
        return False
    return True


def _start_arecord(device: str, rate: int):
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-c", "1",
        "-r", str(rate),
        "-t", "raw",
        "-q",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )


def _rms_pcm16(data: bytes) -> float:
    if not data:
        return 0.0
    try:
        import array
        a = array.array("h")
        a.frombytes(data)
        if not a:
            return 0.0
        # RMS = sqrt(mean(x^2))
        s = 0.0
        for v in a:
            s += float(v) * float(v)
        return (s / float(len(a)) + 1e-9) ** 0.5
    except Exception:
        return 0.0


class AudioPlayer:
    def __init__(self, volume: int = 80):
        self._lock = Lock()
        self._music = None
        if Music is not None:
            try:
                self._music = Music()
                self._music.music_set_volume(int(volume))
            except Exception:
                self._music = None

    def _get_audio_duration_sec(self, filepath: str):
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

    def _play_with_music(self, filepath: str):
        dur = self._get_audio_duration_sec(filepath)
        if dur is None:
            dur = 2.5
        with self._lock:
            self._music.music_play(str(filepath), loops=1)
        time.sleep(max(0.1, dur + 0.15))

    def _play_with_fallback(self, filepath: str):
        if shutil.which("mpg123"):
            subprocess.run(["mpg123", "-q", filepath], check=False)
            return
        if shutil.which("aplay"):
            subprocess.run(["aplay", "-q", filepath], check=False)
            return
        if shutil.which("ffplay"):
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", filepath], check=False)
            return
        print("ERROR: no audio player found (robot_hat/Music, mpg123, ffplay)", flush=True)

    def play_mp3(self, filepath: str):
        if self._music is not None:
            try:
                self._play_with_music(filepath)
                return
            except Exception as e:
                print("[PLAY] music_play error:", e, flush=True)
        self._play_with_fallback(filepath)

    def play_wav(self, filepath: str):
        if self._music is not None:
            try:
                self._play_with_music(filepath)
                return
            except Exception as e:
                print("[PLAY] music_play error:", e, flush=True)
        self._play_with_fallback(filepath)


def _boot_pidog_if_available(pose_file: str):
    if MotionController is None:
        print("[PIDOG] MotionController not available", flush=True)
        return None
    try:
        mc = MotionController(pose_file=Path(pose_file))
        print("[PIDOG] boot...", flush=True)
        mc.boot()
        time.sleep(0.8)
        return mc
    except Exception as e:
        print("[PIDOG] boot error:", e, flush=True)
        return None


def _post_request(url: str, text: str, req_id: str):
    r = None
    try:
        payload = {"text": text}
        if req_id:
            payload["id"] = req_id
        params = {"format": "json"}
        print("[HTTP] post ->", url, "id=", req_id, flush=True)
        r = requests.post(url, params=params, json=payload, timeout=20)
        print("[HTTP] status=", r.status_code, flush=True)
        if r.status_code != 200:
            return None
        data = r.json()
        print("[HTTP] response=", data, flush=True)
        if isinstance(data, dict) and data.get("ok") is True:
            return data
    except Exception as e:
        body = ""
        try:
            body = (r.text or "").strip() if r is not None else ""
        except Exception:
            body = ""
        if body:
            print("[HTTP] response text=", body[:200], flush=True)
        print("[HTTP] post error:", e, flush=True)
    return None


def _save_mp3_response(resp, req_id: str):
    tmpdir = tempfile.mkdtemp(prefix="pidog_ans_")
    out = os.path.join(tmpdir, f"ans_{req_id}.mp3")
    with open(out, "wb") as f:
        for chunk in resp.iter_content(chunk_size=16384):
            if not chunk:
                continue
            f.write(chunk)
    if os.path.getsize(out) < 1024:
        print("[HTTP] empty mp3", flush=True)
        return None
    return out


def _download_mp3_from_url(url: str, req_id: str):
    try:
        r = requests.get(url, timeout=45, stream=True)
        if r.status_code != 200:
            print("[HTTP] mp3 status=", r.status_code, flush=True)
            return None
        return _save_mp3_response(r, req_id)
    except Exception as e:
        print("[HTTP] mp3 error:", e, flush=True)
        return None


def _get_status_json(url: str, req_id: str):
    try:
        params = {"id": req_id, "format": "json"}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            print("[HTTP] status status_code=", r.status_code, flush=True)
            return None
        raw_text = (r.text or "").strip()
        if raw_text:
            print("[HTTP] raw body=", raw_text[:500], flush=True)
        data = r.json()
        if isinstance(data, dict):
            status = (data.get("status") or "").strip()
            if status:
                print("[STATUS] http id=", req_id, "status=", status, flush=True)
            return data
        return None
    except Exception as e:
        print("[HTTP] status error:", e, flush=True)
        return None


def _wait_status_done(status_url: str, req_id: str, timeout_sec: float, interval_sec: float):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        data = _get_status_json(status_url, req_id)
        if isinstance(data, dict):
            status = (data.get("status") or "").strip().lower()
            audio_url = (data.get("audio_url") or "").strip()
            warning = (data.get("warning") or "").strip()
            error_msg = (data.get("error") or "").strip()
            if warning:
                print("[STATUS] warning=", warning, flush=True)
            if status == "error":
                print("[STATUS] error=", error_msg or "unknown", flush=True)
                return False
            if audio_url:
                mp3_path = _download_mp3_from_url(audio_url, req_id)
                if mp3_path:
                    return mp3_path
            if status == "done" and not audio_url:
                print("[STATUS] done but audio_url missing, keep polling", flush=True)
        time.sleep(interval_sec)
    return False


def main():
    parser = argparse.ArgumentParser(description="Distributed listening client (VOSK + MQTT)")
    parser.add_argument("--model", default=os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-vn-0.4"))
    parser.add_argument("--device", default=os.environ.get("VOSK_MIC_DEVICE", "default"))
    parser.add_argument("--rate", type=int, default=int(os.environ.get("VOSK_SAMPLE_RATE", "16000")))
    parser.add_argument("--chunk", type=int, default=int(os.environ.get("VOSK_CHUNK_BYTES", "8000")))
    parser.add_argument("--request-url", default=os.environ.get(
        "REQUEST_URL",
        "https://embeddedprogramming-healtheworldserver.up.railway.app/pidog/chat/request",
    ))
    parser.add_argument("--status-url", default=os.environ.get(
        "STATUS_URL",
        "https://embeddedprogramming-healtheworldserver.up.railway.app/pidog/chat/status",
    ))
    parser.add_argument("--status-interval", type=float, default=float(os.environ.get("STATUS_POLL_SEC", "1.5")))
    parser.add_argument("--status-timeout", type=float, default=float(os.environ.get("STATUS_TIMEOUT_SEC", "90")))
    parser.add_argument("--volume", type=int, default=int(os.environ.get("PLAY_VOLUME", "80")))
    parser.add_argument("--waiting-wav", default=os.environ.get("WAITING_WAV", "waitingmessage.wav"))
    parser.add_argument("--pidog-pose-file", default=os.environ.get("PIDOG_POSE_FILE", "pidog_pose_config.txt"))
    parser.add_argument("--min-rms", type=float, default=float(os.environ.get("MIN_RMS", "650")))
    parser.add_argument("--min-words", type=int, default=int(os.environ.get("MIN_WORDS", "3")))
    parser.add_argument("--min-chars", type=int, default=int(os.environ.get("MIN_CHARS", "6")))
    parser.add_argument("--wait-before-request", type=float, default=float(os.environ.get("WAIT_BEFORE_REQUEST_SEC", "1.5")))
    args = parser.parse_args()

    if not _require_arecord():
        return 2

    if not os.path.isdir(args.model):
        print("ERROR: model folder not found:", args.model, flush=True)
        return 2

    try:
        os.system("pinctrl set 12 op dh")
        time.sleep(0.1)
    except Exception:
        pass

    mc = _boot_pidog_if_available(args.pidog_pose_file)

    print("[INIT] loading model:", args.model, flush=True)
    model = Model(args.model)
    rec = KaldiRecognizer(model, args.rate)
    rec.SetWords(True)

    player = AudioPlayer(volume=args.volume)
    waiting_path = Path(args.waiting_wav)

    print("[MIC] device=", args.device, "rate=", args.rate, "chunk=", args.chunk, flush=True)
    print("[HTTP] request=", args.request_url, flush=True)
    print("[HTTP] status=", args.status_url, flush=True)
    print("[RUN] listening... (Ctrl+C to stop)", flush=True)
    print("[GATE] min_rms=", args.min_rms, "min_words=", args.min_words, "min_chars=", args.min_chars, flush=True)

    proc = None
    busy = False
    playing = False
    last_voice_ts = 0.0

    try:
        proc = _start_arecord(args.device, args.rate)
        if proc.stdout is None:
            print("ERROR: failed to open arecord stdout", flush=True)
            return 2

        while True:
            data = proc.stdout.read(args.chunk)
            if not data:
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue

            if busy or playing:
                # keep draining audio but ignore during request
                continue

            if args.min_rms > 0:
                rms = _rms_pcm16(data)
                if rms >= float(args.min_rms):
                    last_voice_ts = time.time()

            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get("text", "").strip()
                if not text:
                    continue
                if args.min_rms > 0 and (time.time() - last_voice_ts) > 1.0:
                    print("[SKIP] final without recent voice", flush=True)
                    continue
                if args.min_chars > 0 and len(text) < int(args.min_chars):
                    print("[SKIP] short text:", text, flush=True)
                    continue
                if args.min_words > 0 and len(text.split()) < int(args.min_words):
                    print("[SKIP] few words:", text, flush=True)
                    continue

                req_id = uuid.uuid4().hex
                print("[FINAL]", text, flush=True)
                print("[REQ] id=", req_id, flush=True)

                if waiting_path.exists():
                    print("[WAIT] play waitingmessage.wav", flush=True)
                    player.play_wav(str(waiting_path))
                else:
                    print("[WAIT] missing waiting wav:", waiting_path, flush=True)

                if args.wait_before_request > 0:
                    time.sleep(float(args.wait_before_request))
                print("[REQ] send transcript -> HTTP /pidog/chat/request", flush=True)
                resp = _post_request(args.request_url, text, req_id)
                if not resp:
                    print("[REQ] HTTP request failed", flush=True)
                    continue

                server_id = (resp.get("id") or resp.get("Id") or "").strip() if isinstance(resp, dict) else ""
                if server_id and server_id != req_id:
                    print("[REQ] server id=", server_id, "override local id", flush=True)
                    req_id = server_id

                audio_url = (resp.get("audio_url") or "").strip() if isinstance(resp, dict) else ""
                if audio_url:
                    mp3_path = _download_mp3_from_url(audio_url, req_id)
                    if mp3_path:
                        print("[PLAY]", mp3_path, flush=True)
                        playing = True
                        player.play_mp3(mp3_path)
                        playing = False
                        continue

                busy = True
                mp3_path = _wait_status_done(args.status_url, req_id, args.status_timeout, args.status_interval)
                busy = False

                if not mp3_path:
                    print("[STATUS] timeout for id=", req_id, flush=True)
                    continue

                print("[PLAY]", mp3_path, flush=True)
                playing = True
                player.play_mp3(mp3_path)
                playing = False
            else:
                res = json.loads(rec.PartialResult())
                partial = res.get("partial", "").strip()
                if partial:
                    print("[PARTIAL]", partial, flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()
        try:
            if mc:
                mc.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
