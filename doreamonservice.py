#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import socket
import sys
import time

import pygame

# Service defaults for desktop
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
if sys.platform != "win32" and not os.environ.get("SDL_VIDEODRIVER"):
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")

BG = (255, 255, 255)
VALID_EMOS = {
    "love_eyes",
    "music",
    "what_is_it",
    "suprise",
    "sleep",
    "sad",
    "angry",
}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_size(value):
    try:
        w_str, h_str = value.lower().split("x")
        return int(w_str), int(h_str)
    except Exception as exc:
        raise ValueError("size must be like 800x600") from exc


def scale_to_fit(surface, target_size):
    tw, th = target_size
    iw, ih = surface.get_size()
    if iw <= 0 or ih <= 0:
        return surface
    scale = min(float(tw) / iw, float(th) / ih)
    new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
    return pygame.transform.smoothscale(surface, new_size)


def list_image_files(folder):
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(path)
    return files


def find_label_file(files, label):
    label_lower = label.lower()
    for path in files:
        base = os.path.basename(path).lower()
        if label_lower in base:
            return path
    return None


def find_talk_files(files):
    indexed = []
    for path in files:
        base = os.path.basename(path).lower()
        for i in range(1, 5):
            if ("talk%d" % i) in base or ("%d_talk" % i) in base or ("talk-%d" % i) in base:
                indexed.append((i, path))
                break
    if indexed:
        indexed.sort(key=lambda item: item[0])
        return [p for _, p in indexed]

    # fallback: any file containing "talk" in name
    fallback = [p for p in files if "talk" in os.path.basename(p).lower()]
    return fallback


def load_asset(path, screen_size):
    surf = pygame.image.load(path).convert_alpha()
    scaled = scale_to_fit(surf, screen_size)
    rect = scaled.get_rect(center=(screen_size[0] // 2, screen_size[1] // 2))
    return scaled, rect


def blit_with_alpha(screen, surface, rect, alpha):
    surface.set_alpha(alpha)
    screen.blit(surface, rect)
    surface.set_alpha(None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Doraemon emotion service using image assets."
    )
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--size", default="800x600")
    parser.add_argument("--port", type=int, default=39393)
    parser.add_argument("--talk-interval", type=float, default=2.0)
    parser.add_argument("--fade", type=float, default=0.2)
    parser.add_argument("--talk-overlay-alpha", type=int, default=60)
    parser.add_argument("--talk-overlay-duration", type=float, default=0.12)
    parser.add_argument("--fps", type=int, default=60)
    return parser.parse_args()


def log(msg):
    print("[doreamonservice] %s" % msg, flush=True)


def main():
    args = parse_args()

    pygame.init()

    if args.windowed:
        w, h = parse_size(args.size)
        screen = pygame.display.set_mode((w, h))
    else:
        flags = pygame.FULLSCREEN | pygame.NOFRAME
        try:
            screen = pygame.display.set_mode((0, 0), flags)
        except pygame.error:
            w, h = parse_size(args.size)
            screen = pygame.display.set_mode((w, h))
            log("Fullscreen failed, fallback to windowed %dx%d" % (w, h))

    pygame.display.set_caption("Doraemon Service")
    pygame.mouse.set_visible(False)
    log("SDL_VIDEODRIVER=%s" % os.environ.get("SDL_VIDEODRIVER", "default"))
    log("Port=%d | windowed=%s | size=%s | talk_interval=%.2f | fade=%.2f" % (
        args.port, args.windowed, args.size, args.talk_interval, args.fade
    ))

    files = list_image_files(os.getcwd())
    if not files:
        log("No image files found in the current folder.")
        return 2

    screen_size = screen.get_size()
    emo_files = {}
    missing = []
    for emo in sorted(VALID_EMOS):
        path = find_label_file(files, emo)
        if not path:
            missing.append(emo)
        else:
            emo_files[emo] = path

    talk_files = find_talk_files(files)

    if missing:
        log("Missing image for labels:")
        for emo in missing:
            log("- %s" % emo)
        log("Ensure filenames include the label text.")
        return 2

    if not talk_files:
        log("Warning: no talk images found (expected names like talk1..talk4).")
    else:
        log("Talk frames: %d" % len(talk_files))

    for emo, path in sorted(emo_files.items()):
        log("Asset %s -> %s" % (emo, os.path.basename(path)))

    assets = {}
    for emo, path in emo_files.items():
        assets[emo] = load_asset(path, screen_size)

    talk_assets = []
    for path in talk_files:
        talk_assets.append(load_asset(path, screen_size))

    # UDP receive commands
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.setblocking(False)
    log("Listening UDP on 127.0.0.1:%d" % args.port)

    emo = "what_is_it"
    mouth_target = 0.0
    mouth_level = 0.0
    last_mouth_ts = 0.0
    force_talk = False
    auto_talk = True

    current_asset = assets.get(emo)
    next_asset = None
    transition_start = 0.0

    talk_idx = 0
    next_talk_switch = 0.0
    talk_overlay_until = 0.0

    clock = pygame.time.Clock()

    last_logged_emo = emo
    last_logged_talk = force_talk
    last_logged_auto_talk = auto_talk

    while True:
        now = time.time()

        # receive cmd
        try:
            data, _ = sock.recvfrom(2048)
            cmd = data.decode("utf-8", "ignore").strip()
            parts = cmd.split()

            if len(parts) == 1:
                if parts[0] in VALID_EMOS:
                    emo = parts[0]
                    auto_talk = False
                elif parts[0].upper() == "TALK":
                    force_talk = True
            elif len(parts) >= 2:
                if parts[0].upper() == "EMO" and parts[1] in VALID_EMOS:
                    emo = parts[1]
                    auto_talk = False
                elif parts[0].upper() == "MOUTH":
                    try:
                        mouth_target = clamp(float(parts[1]), 0.0, 1.0)
                        last_mouth_ts = now
                    except Exception:
                        pass
                elif parts[0].upper() == "TALK":
                    force_talk = parts[1] not in ("0", "off", "false")
                    if force_talk:
                        auto_talk = True
        except BlockingIOError:
            pass

        # decay mouth
        if last_mouth_ts > 0 and (now - last_mouth_ts) > 0.25:
            mouth_target = 0.0

        mouth_level = mouth_level + (mouth_target - mouth_level) * 0.22

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return 0
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return 0

        is_talking = auto_talk or (mouth_level > 0.02) or force_talk
        if emo != last_logged_emo:
            log("EMO -> %s" % emo)
            last_logged_emo = emo
        if force_talk != last_logged_talk:
            log("TALK -> %s" % ("on" if force_talk else "off"))
            last_logged_talk = force_talk
        if auto_talk != last_logged_auto_talk:
            log("AUTO_TALK -> %s" % ("on" if auto_talk else "off"))
            last_logged_auto_talk = auto_talk

        if is_talking and talk_assets:
            if now >= next_talk_switch:
                talk_idx = (talk_idx + 1) % len(talk_assets)
                next_talk_switch = now + max(0.02, args.talk_interval)
                talk_overlay_until = now + max(0.0, args.talk_overlay_duration)
            desired_asset = talk_assets[talk_idx]
        else:
            desired_asset = assets.get(emo)

        if current_asset is None:
            current_asset = desired_asset

        if desired_asset != current_asset and desired_asset is not None:
            if is_talking:
                current_asset = desired_asset
                next_asset = None
            else:
                if next_asset != desired_asset:
                    next_asset = desired_asset
                    transition_start = now
        if is_talking:
            next_asset = None

        screen.fill(BG)

        if next_asset is not None:
            alpha = clamp((now - transition_start) / max(0.01, args.fade), 0.0, 1.0)
            if current_asset:
                blit_with_alpha(screen, current_asset[0], current_asset[1], int(255 * (1.0 - alpha)))
            blit_with_alpha(screen, next_asset[0], next_asset[1], int(255 * alpha))
            if alpha >= 1.0:
                current_asset = next_asset
                next_asset = None
        else:
            if current_asset:
                screen.blit(current_asset[0], current_asset[1])
        if is_talking and current_asset and now < talk_overlay_until and args.talk_overlay_alpha > 0:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, clamp(args.talk_overlay_alpha, 0, 255)))
            screen.blit(overlay, (0, 0))

        pygame.display.flip()
        clock.tick(args.fps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
