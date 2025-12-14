#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import ssl
import paho.mqtt.client as mqtt

MQTT_HOST = "rfff7184.ala.us-east-1.emqxsl.com"
MQTT_PORT = 8883
MQTT_USER = "robot_matthew"
MQTT_PASS = "29061992abCD!yesokmen"

TOPIC = "/pidog/sensorhubdata"

last_ts = None

def on_connect(client, userdata, flags, rc, properties=None):
    print("[MQTT] connected rc =", rc)
    client.subscribe(TOPIC, qos=0)
    print("[MQTT] subscribed:", TOPIC)

def on_message(client, userdata, msg):
    global last_ts
    ts = time.strftime("%H:%M:%S")

    try:
        payload = msg.payload.decode("utf-8", errors="ignore").strip()
        data = json.loads(payload)

        temp = data.get("temp_c", None)
        humid = data.get("humid", None)
        dist = data.get("distance_cm", None)
        ts_ms = data.get("ts_ms", None)

        # latency estimate (nếu board gửi millis, chỉ xem "delta giữa các gói")
        if last_ts is None:
            delta = "NA"
        else:
            delta = f"{(ts_ms - last_ts)} ms" if (ts_ms is not None and last_ts is not None) else "NA"
        last_ts = ts_ms if ts_ms is not None else last_ts

        print(f"[{ts}] temp={temp}C  humid={humid}%  dist={dist}cm  dt={delta}")

    except Exception as e:
        print(f"[{ts}] RAW:", msg.payload[:200], "ERR:", e)

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    client.username_pw_set(MQTT_USER, MQTT_PASS)

    # TLS nhưng insecure (không verify cert)
    client.tls_set(cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)

    client.on_connect = on_connect
    client.on_message = on_message

    print("=== MQTT SENSORHUB TEST (Ctrl+C để dừng) ===")
    print("Broker:", MQTT_HOST, "Port:", MQTT_PORT)
    print("Topic:", TOPIC)
    print("------------------------------------------")

    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_forever()

if __name__ == "__main__":
    main()
