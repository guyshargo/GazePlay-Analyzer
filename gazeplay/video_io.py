"""Video probing and fast sequential frame sampling."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    def to_dict(self):
        return {"fps": round(self.fps, 3), "frames": self.n_frames, "width": self.width,
                "height": self.height, "duration_s": round(self.duration, 3)}


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frames from video: {path}")
    h, w = frame.shape[:2]  # honours rotation metadata
    if fps <= 1 or fps > 240:
        fps = 25.0
    if n <= 0:
        n = int(fps * 3600)  # unknown length; reader stops at the real end
    return VideoInfo(str(path), fps, n, w, h)


class SampledReader:
    """Return frames at increasing timestamps without random seeking.

    Skipped frames are only grabbed (decoded, not converted), which is several
    times faster than seeking for every sample. Large forward jumps seek.
    """

    def __init__(self, path: str, info: VideoInfo | None = None):
        self.info = info or probe_video(path)
        self.cap = cv2.VideoCapture(str(path))
        self.next_idx = 0
        self.finished = False
        self._last = None

    def read_at(self, t: float, need_frame: bool = True):
        if self.finished or t < 0:
            return None
        idx = int(round(t * self.info.fps))
        if idx >= self.info.n_frames:
            return None
        if idx == self.next_idx - 1:                      # same frame asked again
            if self._last is not None or not need_frame:
                return self._last if need_frame else None
        if idx < self.next_idx or idx - self.next_idx > self.info.fps * 20:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.next_idx = idx
        while self.next_idx < idx:
            if not self.cap.grab():
                self.finished = True
                return None
            self.next_idx += 1
        if need_frame:
            ok, frame = self.cap.read()
        else:
            ok, frame = self.cap.grab(), None
        self.next_idx += 1
        if not ok:
            self.finished = True
            return None
        self._last = frame
        return frame

    def close(self):
        self.cap.release()


def letterbox_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (max(2, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)
