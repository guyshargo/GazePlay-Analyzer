"""Side-by-side review video: room camera | eye-tracking camera, with the final labels."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .sync import ffmpeg_exe
from .timeline import hex_to_bgr

PANEL_H = 360
BANNER_H = 64


class FrameStore:
    """Keeps one small annotated frame per sample in a temporary MJPG file."""

    def __init__(self, path: Path, fps: float, src_w: int, src_h: int, title: str):
        self.path = Path(path)
        self.size = (max(2, int(round(src_w * PANEL_H / src_h)) // 2 * 2), PANEL_H)
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), fps, self.size)
        self.title = title
        self.count = 0

    def add(self, img: np.ndarray | None, note: str | None = None):
        if img is None:
            frame = np.full((self.size[1], self.size[0], 3), 40, np.uint8)
            cv2.putText(frame, note or "no frame", (20, self.size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            frame = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        (tw, _), _ = cv2.getTextSize(self.title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (0, 0), (tw + 16, 26), (30, 30, 30), -1)
        cv2.putText(frame, self.title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.writer.write(frame)
        self.count += 1

    def close(self):
        self.writer.release()


def _chip(img, x, y, text, color_bgr, fg=(255, 255, 255), scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 14, y + 6), color_bgr, -1)
    cv2.putText(img, text, (x + 7, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)
    return x + tw + 14


def compose(room_store: FrameStore, ego_store: FrameStore, out_path: Path, timeline, colors: dict,
            fps: float, log=lambda m: None) -> Path | None:
    cap_r = cv2.VideoCapture(str(room_store.path))
    cap_e = cv2.VideoCapture(str(ego_store.path))
    w = room_store.size[0] + ego_store.size[0]
    h = PANEL_H + BANNER_H
    raw_path = Path(out_path).with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    grey = (110, 110, 110)
    for row in timeline.itertuples(index=False):
        ok1, fr = cap_r.read()
        ok2, fe = cap_e.read()
        if not ok1:
            fr = np.zeros((PANEL_H, room_store.size[0], 3), np.uint8)
        if not ok2:
            fe = np.zeros((PANEL_H, ego_store.size[0], 3), np.uint8)
        banner = np.full((BANNER_H, w, 3), 250, np.uint8)
        ts, inter, gaze, match = row[0], row[1], row[2], row[3]
        cv2.putText(banner, ts, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        x = 190
        cv2.putText(banner, "Playing with:", (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        x = _chip(banner, x + 112, 40, inter, hex_to_bgr(colors[inter]) if inter in colors else grey) + 24
        cv2.putText(banner, "Looking at:", (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        x = _chip(banner, x + 98, 40, gaze, hex_to_bgr(colors[gaze]) if gaze in colors else grey) + 24
        if match == "Yes":
            _chip(banner, max(x, w - 230), 40, "LOOKING AT IT", (64, 160, 43))
        elif match == "No":
            _chip(banner, max(x, w - 230), 40, "LOOKING ELSEWHERE", (60, 60, 200))
        writer.write(np.vstack([np.hstack([fr, fe]), banner]))
    writer.release()
    cap_r.release()
    cap_e.release()

    # Re-encode to H.264 so the browser can play it inline (and files stay small).
    exe = ffmpeg_exe()
    if exe:
        cmd = [exe, "-y", "-v", "error", "-i", str(raw_path), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
            if proc.returncode == 0 and Path(out_path).exists():
                raw_path.unlink(missing_ok=True)
                return Path(out_path)
            log("Could not convert the review video to H.264; kept the MPEG-4 version.")
        except Exception:
            pass
    Path(out_path).unlink(missing_ok=True)
    raw_path.rename(out_path)
    return Path(out_path)
