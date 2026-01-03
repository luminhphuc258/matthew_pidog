#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
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


def scale_by_factor(surface, scale):
    if scale <= 0:
        return surface
    w, h = surface.get_size()
    return pygame.transform.smoothscale(
        surface, (max(1, int(w * scale)), max(1, int(h * scale)))
    )


def maybe_set_colorkey(surface, force=False):
    try:
        c = surface.get_at((0, 0))
    except Exception:
        return False
    if force and c.a == 255:
        surface.set_colorkey(c)
        return True
    if c.a == 255 and c.r >= 245 and c.g >= 245 and c.b >= 245:
        surface.set_colorkey(c)
        return True
    return False


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
    parser.add_argument("--talk-interval", type=float, default=1.2)
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
    face_dir = os.path.join(os.getcwd(), "doreamonface")
    face_files = list_image_files(face_dir) if os.path.isdir(face_dir) else []
    face_missing = []

    def face_path(label, record_missing=True):
        path = find_label_file(face_files, label)
        if not path and record_missing:
            face_missing.append(label)
        return path

    baseface_path = None
    if face_files:
        baseface_path = face_path("facebase", record_missing=False)
        if not baseface_path:
            baseface_path = face_path("facebase", record_missing=False)
        if not baseface_path:
            face_missing.append("facebase")
    eyeopen_path = face_path("eyeopen") if face_files else None
    eyeclose_path = face_path("eyeclose") if face_files else None
    eyeleft_path = face_path("eyeleft") if face_files else None
    eyeright_path = face_path("eyeright") if face_files else None
    mouthopen_path = face_path("mouthopen") if face_files else None
    mouthmedium_path = face_path("mouthmedium") if face_files else None
    mouthclose_path = face_path("mouthclose") if face_files else None
    emo_files = {}
    missing = []
    for emo in sorted(VALID_EMOS):
        path = find_label_file(files, emo)
        if not path:
            missing.append(emo)
        else:
            emo_files[emo] = path

    talk_files = find_talk_files(files)
    mouth_big_path = find_label_file(files, "mouth_big")
    mouth_medium_path = find_label_file(files, "mouth_medium")
    mouth_closed_path = find_label_file(files, "mouth_closed")

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
    mouth_missing = []
    if not mouth_big_path:
        mouth_missing.append("mouth_big")
    if not mouth_medium_path:
        mouth_missing.append("mouth_medium")
    if not mouth_closed_path:
        mouth_missing.append("mouth_closed")
    if mouth_missing:
        log("Warning: missing mouth images for talk sequence: %s" % ", ".join(mouth_missing))

    face_assets = {}
    use_face_parts = False
    if not face_files:
        log("Error: doreamonface folder not found or empty.")
        return 2
    if not baseface_path:
        log("Error: doreamonface missing baseface image.")
        return 2

    base_raw = pygame.image.load(baseface_path).convert_alpha()
    base_scale = min(
        float(screen_size[0]) / base_raw.get_width(),
        float(screen_size[1]) / base_raw.get_height(),
    )
    base_scaled = scale_by_factor(base_raw, base_scale)
    base_rect = base_scaled.get_rect(center=(screen_size[0] // 2, screen_size[1] // 2))

    eye_y_ratio = 0.32
    mouth_y_ratio = 0.68

    def load_face_part(path, label, role):
        if not path:
            return None
        part_raw = pygame.image.load(path).convert_alpha()
        if maybe_set_colorkey(part_raw, force=True):
            log("Doreamonface %s: set colorkey from top-left pixel." % label)
        part_scaled = scale_by_factor(part_raw, base_scale)
        if part_raw.get_size() == base_raw.get_size():
            rect = part_scaled.get_rect(topleft=base_rect.topleft)
        else:
            if role == "eye":
                anchor = (base_rect.centerx, base_rect.top + int(base_scaled.get_height() * eye_y_ratio))
                rect = part_scaled.get_rect(center=anchor)
            elif role == "mouth":
                anchor = (base_rect.centerx, base_rect.top + int(base_scaled.get_height() * mouth_y_ratio))
                rect = part_scaled.get_rect(center=anchor)
            else:
                rect = part_scaled.get_rect(center=base_rect.center)
        return part_scaled, rect

    face_assets["base"] = (base_scaled, base_rect)
    face_assets["eyeopen"] = load_face_part(eyeopen_path, "eyeopen", "eye")
    face_assets["eyeclose"] = load_face_part(eyeclose_path, "eyeclose", "eye")
    face_assets["eyeleft"] = load_face_part(eyeleft_path, "eyeleft", "eye")
    face_assets["eyeright"] = load_face_part(eyeright_path, "eyeright", "eye")
    face_assets["mouthopen"] = load_face_part(mouthopen_path, "mouthopen", "mouth")
    face_assets["mouthmedium"] = load_face_part(mouthmedium_path, "mouthmedium", "mouth")
    face_assets["mouthclose"] = load_face_part(mouthclose_path, "mouthclose", "mouth")
    use_face_parts = bool(
        face_assets["base"]
        and face_assets["eyeopen"]
        and face_assets["eyeclose"]
        and face_assets["eyeleft"]
        and face_assets["eyeright"]
        and face_assets["mouthopen"]
        and face_assets["mouthmedium"]
        and face_assets["mouthclose"]
    )
    if face_missing:
        log("Error: missing doreamonface parts: %s" % ", ".join(sorted(set(face_missing))))
        return 2
    if not use_face_parts:
        log("Error: doreamonface parts incomplete; cannot run.")
        return 2
    log("Talk mode: using doreamonface parts.")
    log("Doreamonface anchor ratios: eyes=%.2f mouth=%.2f" % (eye_y_ratio, mouth_y_ratio))
    log("Doreamonface scale=%.3f | base_raw=%dx%d | base_scaled=%dx%d" % (
        base_scale,
        base_raw.get_width(),
        base_raw.get_height(),
        base_scaled.get_width(),
        base_scaled.get_height(),
    ))
    face_path_map = {
        "baseface": baseface_path,
        "eyeopen": eyeopen_path,
        "eyeclose": eyeclose_path,
        "eyeleft": eyeleft_path,
        "eyeright": eyeright_path,
        "mouthopen": mouthopen_path,
        "mouthmedium": mouthmedium_path,
        "mouthclose": mouthclose_path,
    }
    for key, asset in (
        ("baseface", face_assets["base"]),
        ("eyeopen", face_assets["eyeopen"]),
        ("eyeclose", face_assets["eyeclose"]),
        ("eyeleft", face_assets["eyeleft"]),
        ("eyeright", face_assets["eyeright"]),
        ("mouthopen", face_assets["mouthopen"]),
        ("mouthmedium", face_assets["mouthmedium"]),
        ("mouthclose", face_assets["mouthclose"]),
    ):
        if asset:
            log("Doreamonface %s -> %s (%dx%d)" % (
                key,
                os.path.basename(face_path_map.get(key, "")),
                asset[0].get_width(),
                asset[0].get_height(),
            ))

    for emo, path in sorted(emo_files.items()):
        log("Asset %s -> %s" % (emo, os.path.basename(path)))
    if mouth_big_path:
        log("Mouth big -> %s" % os.path.basename(mouth_big_path))
    if mouth_medium_path:
        log("Mouth medium -> %s" % os.path.basename(mouth_medium_path))
    if mouth_closed_path:
        log("Mouth closed -> %s" % os.path.basename(mouth_closed_path))
    assets = {}
    for emo, path in emo_files.items():
        assets[emo] = load_asset(path, screen_size)

    talk_assets = []
    for path in talk_files:
        talk_assets.append(load_asset(path, screen_size))
    talk_variant_assets = []
    for idx in (2, 3, 4):
        for path in talk_files:
            base = os.path.basename(path).lower()
            if ("talk%d" % idx) in base or ("%d_talk" % idx) in base or ("talk-%d" % idx) in base:
                talk_variant_assets.append(load_asset(path, screen_size))
                break
    mouth_big_asset = load_asset(mouth_big_path, screen_size) if mouth_big_path else None
    mouth_medium_asset = load_asset(mouth_medium_path, screen_size) if mouth_medium_path else None
    mouth_closed_asset = load_asset(mouth_closed_path, screen_size) if mouth_closed_path else None
    use_mouth_sequence = bool(
        mouth_big_asset and mouth_medium_asset and mouth_closed_asset
    )

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
    talk_seq = []
    talk_seq_idx = 0
    next_talk_switch = 0.0
    talk_overlay_until = 0.0
    talk_sequence = []
    talk_seq_pos = 0
    eye_state = "open"
    eye_next_switch = 0.0
    mouth_state = "close"
    mouth_next_switch = 0.0
    talk_pair_seq = []
    talk_pair_idx = 0

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

        talk_face_mode = is_talking and use_face_parts
        desired_asset = None
        if talk_face_mode:
            if now >= mouth_next_switch:
                if not talk_pair_seq or talk_pair_idx >= len(talk_pair_seq):
                    middle_pairs = [
                        ("open", "medium"),
                        ("close", "close"),
                        ("left", "medium"),
                        ("right", "close"),
                    ]
                    random.shuffle(middle_pairs)
                    talk_pair_seq = [("open", "open")] + middle_pairs + [("open", "close")]
                    talk_pair_idx = 0

                eye_state, mouth_state = talk_pair_seq[talk_pair_idx]
                talk_pair_idx += 1

                mouth_next_switch = now + max(0.35, args.talk_interval * 1.1)
                eye_next_switch = mouth_next_switch
        elif is_talking and (talk_assets or use_mouth_sequence):
            if use_mouth_sequence:
                if not talk_sequence:
                    tail_asset = None
                    if talk_variant_assets:
                        tail_asset = random.choice(talk_variant_assets)
                    elif talk_assets:
                        tail_asset = talk_assets[0]
                    else:
                        tail_asset = mouth_closed_asset
                    talk_sequence = [
                        mouth_big_asset,
                        mouth_medium_asset,
                        mouth_closed_asset,
                        tail_asset,
                    ]
                    talk_seq_pos = 0
                    next_talk_switch = now + max(0.02, args.talk_interval)
                    talk_overlay_until = now + max(0.0, args.talk_overlay_duration)
                elif now >= next_talk_switch:
                    talk_seq_pos += 1
                    if talk_seq_pos >= len(talk_sequence):
                        tail_asset = None
                        if talk_variant_assets:
                            tail_asset = random.choice(talk_variant_assets)
                        elif talk_assets:
                            tail_asset = talk_assets[0]
                        else:
                            tail_asset = mouth_closed_asset
                        talk_sequence = [
                            mouth_big_asset,
                            mouth_medium_asset,
                            mouth_closed_asset,
                            tail_asset,
                        ]
                        talk_seq_pos = 0
                    next_talk_switch = now + max(0.02, args.talk_interval)
                    talk_overlay_until = now + max(0.0, args.talk_overlay_duration)
                desired_asset = talk_sequence[talk_seq_pos]
            else:
                if not talk_seq:
                    talk_seq = [0]
                    talk_seq_idx = 0
                    talk_idx = talk_seq[talk_seq_idx]
                    next_talk_switch = now + max(0.02, args.talk_interval)
                    talk_overlay_until = now + max(0.0, args.talk_overlay_duration)
                elif now >= next_talk_switch:
                    talk_seq_idx += 1
                    if talk_seq_idx >= len(talk_seq):
                        talk_seq_idx = 0
                    talk_idx = talk_seq[talk_seq_idx]
                    next_talk_switch = now + max(0.02, args.talk_interval)
                    talk_overlay_until = now + max(0.0, args.talk_overlay_duration)
                desired_asset = talk_assets[talk_idx]
        else:
            desired_asset = assets.get(emo)

        if current_asset is None:
            current_asset = desired_asset

        if talk_face_mode:
            next_asset = None
        elif desired_asset != current_asset and desired_asset is not None:
            if is_talking:
                current_asset = desired_asset
                next_asset = None
            else:
                if next_asset != desired_asset:
                    next_asset = desired_asset
                    transition_start = now
        if is_talking and not talk_face_mode:
            next_asset = None

        screen.fill(BG)

        if talk_face_mode and face_assets:
            base_asset = face_assets.get("base")
            if base_asset:
                screen.blit(base_asset[0], base_asset[1])
            eye_key = {
                "open": "eyeopen",
                "close": "eyeclose",
                "left": "eyeleft",
                "right": "eyeright",
            }.get(eye_state, "eyeopen")
            mouth_key = {
                "open": "mouthopen",
                "medium": "mouthmedium",
                "close": "mouthclose",
            }.get(mouth_state, "mouthclose")
            eye_asset = face_assets.get(eye_key)
            mouth_asset = face_assets.get(mouth_key)
            if eye_asset:
                screen.blit(eye_asset[0], eye_asset[1])
            if mouth_asset:
                screen.blit(mouth_asset[0], mouth_asset[1])
        else:
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
