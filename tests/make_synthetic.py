"""Generate a synthetic session: room video, eye-tracking video (both with sound), Tobii-style gaze TSV
and the ground truth, for end-to-end testing without real recordings or YOLO models.

Script (eye-tracking time):  0-20 s ball in hand, 20-40 block, 40-55 toy car, 55-70 hands free.
Gaze:                        0-20 ball (10-12 background), 20-40 ball, 40-55 car, 55-70 block.
The room camera starts 7.3 s before the glasses.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gazeplay.sync import ffmpeg_exe  # noqa: E402

FPS = 25
OFFSET = 7.3
EGO_DUR = 70.0
ROOM_DUR = 80.0
SR = 16000
RED, BLUE, GREEN = (40, 40, 220), (220, 60, 30), (40, 190, 40)
MAGENTA, YELLOW, CYAN = (200, 40, 200), (30, 220, 230), (220, 210, 40)


def held_object(te: float):
    if 0 <= te < 20:
        return "ball"
    if 20 <= te < 40:
        return "block"
    if 40 <= te < 55:
        return "toy car"
    return None


def gaze_target(te: float):
    if 10 <= te < 12:
        return "background"
    if te < 40:
        return "ball"
    if te < 55:
        return "toy car"
    return "block"


def blink(te: float):
    return 30.0 <= te < 30.6 or 61.0 <= te < 62.0


REST = {"ball": (380, 470), "block": (620, 470), "toy car": (520, 560)}
HOLD = {"ball": ((420, 445), (440, 385)), "block": ((585, 450), (560, 390)), "toy car": ((520, 500), (515, 430))}


def draw_object(img, name, c, scale=1.0):
    x, y = int(c[0]), int(c[1])
    if name == "ball":
        cv2.circle(img, (x, y), int(22 * scale), RED, -1)
    elif name == "block":
        s = int(20 * scale)
        cv2.rectangle(img, (x - s, y - s), (x + s, y + s), BLUE, -1)
    else:
        w, h = int(30 * scale), int(15 * scale)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), GREEN, -1)


def room_frame(tr: float):
    img = np.full((720, 1280, 3), 205, np.uint8)
    cv2.rectangle(img, (0, 0), (1280, 230), (225, 225, 225), -1)  # wall
    # adult (larger person, static)
    cv2.rectangle(img, (1050, 250), (1150, 450), MAGENTA, -1)
    for p in ((1040, 470), (1160, 470)):
        cv2.circle(img, p, 6, YELLOW, -1)
    for p in ((1045, 380), (1155, 380)):
        cv2.circle(img, p, 6, CYAN, -1)
    # child
    sway = 6 * np.sin(tr * 0.8)
    x1, y1 = int(470 + sway), 330
    cv2.rectangle(img, (x1, y1), (x1 + 60, y1 + 90), MAGENTA, -1)
    held = held_object(tr - OFFSET)
    wobble = np.array([4 * np.sin(tr * 3), 3 * np.cos(tr * 2.5)])
    left_w, left_e = np.array((460 + sway, 410)), np.array((455 + sway, 370))
    right_w, right_e = np.array((540 + sway, 410)), np.array((545 + sway, 370))
    for name, pos in REST.items():
        if name != held:
            draw_object(img, name, pos)
    if held:
        w, e = HOLD[held]
        w = np.array(w) + wobble
        if held == "ball":
            left_w, left_e = w, np.array(e)
        else:
            right_w, right_e = w, np.array(e)
        hand = w + 0.3 * (w - np.array(e))
        draw_object(img, held, hand)
    for p in (left_w, right_w):
        cv2.circle(img, (int(p[0]), int(p[1])), 6, YELLOW, -1)
    for p in (left_e, right_e):
        cv2.circle(img, (int(p[0]), int(p[1])), 6, CYAN, -1)
    return img


EGO_POS = {"ball": (250, 300), "block": (520, 260), "toy car": (420, 430), "background": (840, 90)}


def ego_drift(te):
    return np.array([40 * np.sin(0.3 * te), 25 * np.sin(0.21 * te)])


def ego_frame(te: float):
    img = np.full((540, 960, 3), 170, np.uint8)
    d = ego_drift(te)
    for name in ("ball", "block", "toy car"):
        draw_object(img, name, np.array(EGO_POS[name]) + d, scale=1.8)
    return img


def write_video(path: Path, frames_fn, dur, size):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    for i in range(int(dur * FPS)):
        vw.write(frames_fn(i / FPS))
    vw.release()


def make_audio(rng):
    n = int(ROOM_DUR * SR)
    base = np.zeros(n, np.float32)
    t = 0.3
    while t < ROOM_DUR - 1:
        dur = rng.uniform(0.05, 0.35)
        a, b = int(t * SR), int((t + dur) * SR)
        burst = rng.normal(0, 1, b - a).astype(np.float32)
        env = np.sin(np.linspace(0, np.pi, b - a)) ** 0.5
        base[a:b] += burst * env * rng.uniform(0.1, 0.5)
        if rng.random() < 0.1:  # clap / toy knock
            base[a:a + 200] += rng.normal(0, 0.9, 200).astype(np.float32)
        t += dur + rng.uniform(0.1, 0.9)
    room = base + rng.normal(0, 0.02, n).astype(np.float32)
    s0 = int(OFFSET * SR)
    ego = base[s0: s0 + int(EGO_DUR * SR)].copy()
    ego = np.convolve(ego, np.ones(5) / 5, mode="same") * 0.6          # different mic: duller, quieter
    ego += 0.03 * np.sin(2 * np.pi * 120 * np.arange(len(ego)) / SR)  # hum
    ego += rng.normal(0, 0.03, len(ego))
    return room, ego.astype(np.float32)


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def mux(video, audio, out):
    subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(out)], check=True)


def make_gaze_tsv(path: Path, rng):
    rows = ["Recording timestamp [ms]\tParticipant name\tRecording name\tEvent\tGaze point X [DACS px]\t"
            "Gaze point Y [DACS px]\tEye movement type\tValidity left"]
    rows.append("0\tP01\tRecording1\tRecordingStart\t\t\t\t")
    t = 0.0
    while t < EGO_DUR:
        ts = f"{t * 1000:.3f}".replace(".", ",")
        if blink(t):
            rows.append(f"{ts}\tP01\tRecording1\t\t\t\tEyesNotFound\tInvalid")
        else:
            p = np.array(EGO_POS[gaze_target(t)]) + ego_drift(t) + rng.normal(0, 5, 2)
            x, y = 2 * p  # exported in the 1920x1080 scene camera space
            rows.append(f"{ts}\tP01\tRecording1\t\t{x:.1f}\t{y:.1f}\tFixation\tValid".replace(".", ","))
        t += 0.02
    path.write_text("\n".join(rows) + "\n", encoding="utf-8-sig")


LOOK_DOWN = (64.0, 66.0)  # Glasses 3 file only: gaze below the scene camera (y > 1)


def make_gazedata_gz(path: Path, rng):
    """Tobii Pro Glasses 3 native format: JSON lines, seconds, normalised gaze2d, 100 Hz."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        t = 0.0013
        while t < EGO_DUR:
            if blink(t):
                rec = {"type": "gaze", "timestamp": round(t, 6), "data": {}}
            else:
                p = np.array(EGO_POS[gaze_target(t)]) + ego_drift(t) + rng.normal(0, 5, 2)
                x, y = p[0] / 960, p[1] / 540
                if LOOK_DOWN[0] <= t < LOOK_DOWN[1]:
                    y = 1.12
                rec = {"type": "gaze", "timestamp": round(t, 6), "data": {
                    "gaze2d": [x, y], "gaze3d": [0, 0, 600],
                    "eyeleft": {"pupildiameter": 4.8}, "eyeright": {"pupildiameter": 4.9}}}
            fh.write(json.dumps(rec) + "\n")
            t += 0.01


def ground_truth(rate=5.0, glasses3=False):
    out = []
    for k in range(int(EGO_DUR * rate)):
        te = k / rate
        h = held_object(te)
        g = gaze_target(te)
        gaze = "No gaze data" if blink(te) else ("No object" if g == "background" else g)
        if glasses3 and LOOK_DOWN[0] <= te < LOOK_DOWN[1]:
            gaze = "Outside camera view"
        out.append((te, h or "No object in hand", gaze))
    return out


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    room_audio, ego_audio = make_audio(rng)
    write_video(out_dir / "_room.mp4", room_frame, ROOM_DUR, (1280, 720))
    write_video(out_dir / "_ego.mp4", ego_frame, EGO_DUR, (960, 540))
    write_wav(out_dir / "_room.wav", room_audio)
    write_wav(out_dir / "_ego.wav", ego_audio)
    mux(out_dir / "_room.mp4", out_dir / "_room.wav", out_dir / "room_camera.mp4")
    mux(out_dir / "_ego.mp4", out_dir / "_ego.wav", out_dir / "tobii_scene.mp4")
    for f in ("_room.mp4", "_ego.mp4", "_room.wav", "_ego.wav"):
        (out_dir / f).unlink()
    make_gaze_tsv(out_dir / "gaze_export.tsv", rng)
    make_gazedata_gz(out_dir / "gazedata.gz", rng)
    print("written to", out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "synthetic")
