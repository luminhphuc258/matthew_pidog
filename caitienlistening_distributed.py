#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from threading import Event, Lock

import requests
from vosk import Model, KaldiRecognizer

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

try:
    from robot_hat import Music
except Exception:
    Music = None


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


class ChatMqttClient:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        client_id: str,
        tls_insecure: bool = True,
    ):
        if mqtt is None:
            raise RuntimeError("paho-mqtt not installed")

        self._host = host
        self._port = int(port)
        self._user = user
        self._password = password
        self._client_id = client_id
        self._tls_insecure = bool(tls_insecure)

        self._events = {}
        self._lock = Lock()
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self._client_id, clean_session=True)
        if self._user:
            self._client.username_pw_set(self._user, self._password)

        if self._tls_insecure:
            self._client.tls_set(cert_reqs=ssl.CERT_NONE)
            self._client.tls_insecure_set(True)

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print("[MQTT] connected rc=", rc, flush=True)

    def _signal_done(self, req_id: str):
        if not req_id:
            return
        with self._lock:
            ev = self._events.get(req_id)
        if ev:
            ev.set()

    def _on_message(self, client, userdata, msg):
        topic = msg.topic or ""
        payload = msg.payload.decode("utf-8", errors="ignore").strip()

        req_id = ""
        status = ""

        # try topic format: /pidog/chat/status/<id>
        if topic.startswith("/pidog/chat/status/"):
            req_id = topic.split("/")[-1].strip()

        if payload:
            try:
                data = json.loads(payload)
                if isinstance(data, dict):
                    req_id = (data.get("id") or data.get("Id") or req_id or "").strip()
                    status = (data.get("status") or data.get("state") or "").strip()
                else:
                    status = str(data).strip()
            except Exception:
                status = payload

        if status.lower() == "done":
            self._signal_done(req_id)

    def connect(self):
        self._client.connect(self._host, self._port, keepalive=30)
        self._client.loop_start()

    def publish_request(self, topic: str, req_id: str, text: str):
        payload = json.dumps({"id": req_id, "text": text}, ensure_ascii=False)
        self._client.publish(topic, payload)

    def wait_for_done(self, status_topic: str, req_id: str, timeout_sec: float):
        ev = Event()
        with self._lock:
            self._events[req_id] = ev

        try:
            self._client.subscribe(status_topic, qos=0)
        except Exception as e:
            print("[MQTT] subscribe error:", e, flush=True)
            return False

        ok = ev.wait(timeout=timeout_sec)
        with self._lock:
            self._events.pop(req_id, None)
        return ok


def _download_mp3(url: str, req_id: str):
    tmpdir = tempfile.mkdtemp(prefix="vosk_ans_")
    out = os.path.join(tmpdir, f"ans_{req_id}.mp3")

    try:
        r = requests.get(url, params={"Id": req_id}, timeout=45, stream=True)
        if r.status_code != 200:
            print("[HTTP] status=", r.status_code, flush=True)
            return None

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "application/json" in ctype:
            try:
                data = r.json()
                print("[HTTP] json:", data, flush=True)
            except Exception:
                print("[HTTP] json response", flush=True)
            return None

        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                f.write(chunk)
        if os.path.getsize(out) < 1024:
            print("[HTTP] empty mp3", flush=True)
            return None
        return out
    except Exception as e:
        print("[HTTP] error:", e, flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Distributed listening client (VOSK + MQTT)")
    parser.add_argument("--model", default=os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-vi-0.4"))
    parser.add_argument("--device", default=os.environ.get("VOSK_MIC_DEVICE", "default"))
    parser.add_argument("--rate", type=int, default=int(os.environ.get("VOSK_SAMPLE_RATE", "16000")))
    parser.add_argument("--chunk", type=int, default=int(os.environ.get("VOSK_CHUNK_BYTES", "8000")))
    parser.add_argument("--mqtt-host", default=os.environ.get("MQTT_HOST", "rfff7184.ala.us-east-1.emqxsl.com"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.environ.get("MQTT_PORT", "8883")))
    parser.add_argument("--mqtt-user", default=os.environ.get("MQTT_USER", "robot_matthew"))
    parser.add_argument("--mqtt-pass", default=os.environ.get("MQTT_PASS", "29061992abCD!yesokmen"))
    parser.add_argument("--mqtt-client-id", default=os.environ.get("MQTT_CLIENT_ID", "pidog-stt-client"))
    parser.add_argument("--mqtt-topic-request", default=os.environ.get("MQTT_TOPIC_REQUEST", "/pidog/chat/request"))
    parser.add_argument("--mqtt-topic-status-prefix", default=os.environ.get("MQTT_TOPIC_STATUS_PREFIX", "/pidog/chat/status"))
    parser.add_argument("--status-timeout", type=float, default=float(os.environ.get("STATUS_TIMEOUT_SEC", "90")))
    parser.add_argument("--answer-url", default=os.environ.get(
        "ANSWER_URL",
        "https://embeddedprogramming-healtheworldserver.up.railway.app/getmyaudioanswer",
    ))
    parser.add_argument("--volume", type=int, default=int(os.environ.get("PLAY_VOLUME", "80")))
    args = parser.parse_args()

    if not _require_arecord():
        return 2

    if not os.path.isdir(args.model):
        print("ERROR: model folder not found:", args.model, flush=True)
        return 2

    if mqtt is None:
        print("ERROR: paho-mqtt not installed", flush=True)
        return 2

    print("[INIT] loading model:", args.model, flush=True)
    model = Model(args.model)
    rec = KaldiRecognizer(model, args.rate)
    rec.SetWords(True)

    mqtt_client = ChatMqttClient(
        host=args.mqtt_host,
        port=args.mqtt_port,
        user=args.mqtt_user,
        password=args.mqtt_pass,
        client_id=args.mqtt_client_id,
        tls_insecure=True,
    )
    mqtt_client.connect()

    player = AudioPlayer(volume=args.volume)

    print("[MIC] device=", args.device, "rate=", args.rate, "chunk=", args.chunk, flush=True)
    print("[MQTT] request=", args.mqtt_topic_request, flush=True)
    print("[MQTT] status prefix=", args.mqtt_topic_status_prefix, flush=True)
    print("[RUN] listening... (Ctrl+C to stop)", flush=True)

    proc = None
    busy = False

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

            if busy:
                # keep draining audio but ignore during request
                continue

            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get("text", "").strip()
                if not text:
                    continue

                req_id = uuid.uuid4().hex
                print("[FINAL]", text, flush=True)
                print("[REQ] id=", req_id, flush=True)

                mqtt_client.publish_request(args.mqtt_topic_request, req_id, text)

                status_topic = f"{args.mqtt_topic_status_prefix}/{req_id}"
                busy = True
                ok = mqtt_client.wait_for_done(status_topic, req_id, args.status_timeout)
                busy = False

                if not ok:
                    print("[STATUS] timeout for id=", req_id, flush=True)
                    continue

                print("[STATUS] done -> fetching audio", flush=True)
                mp3_path = _download_mp3(args.answer_url, req_id)
                if not mp3_path:
                    continue

                print("[PLAY]", mp3_path, flush=True)
                player.play_mp3(mp3_path)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
