#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from vosk import Model, KaldiRecognizer


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


def main():
    parser = argparse.ArgumentParser(description="VOSK mic test via arecord")
    parser.add_argument("--model", default=os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-vn-0.4"))
    parser.add_argument("--device", default=os.environ.get("VOSK_MIC_DEVICE", "default"))
    parser.add_argument("--rate", type=int, default=int(os.environ.get("VOSK_SAMPLE_RATE", "16000")))
    parser.add_argument("--chunk", type=int, default=int(os.environ.get("VOSK_CHUNK_BYTES", "8000")))
    args = parser.parse_args()

    if not _require_arecord():
        return 2

    if not os.path.isdir(args.model):
        print("ERROR: model folder not found:", args.model, flush=True)
        print("Set --model or VOSK_MODEL_PATH to a valid VOSK model directory.", flush=True)
        return 2

    print("[INIT] loading model:", args.model, flush=True)
    model = Model(args.model)
    rec = KaldiRecognizer(model, args.rate)
    rec.SetWords(True)

    print("[MIC] device=", args.device, "rate=", args.rate, "chunk=", args.chunk, flush=True)
    print("[MIC] listening... (Ctrl+C to stop)", flush=True)

    proc = None
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

            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get("text", "").strip()
                if text:
                    print("[FINAL]", text, flush=True)
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
            res = json.loads(rec.FinalResult())
            text = res.get("text", "").strip()
            if text:
                print("[FINAL]", text, flush=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
