"""Eye-tracking (first-person) video: which object is under the gaze point."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import NO_GAZE, NO_OBJECT, PERSON_LABEL
from .detectors import Detection
from .room_analysis import _label, distance_to_detection


@dataclass
class GazeSample:
    t: float
    label: str
    gaze: tuple | None = None          # (x, y) in frame pixels
    movement: str | None = None
    distance_px: float | None = None   # 0 = inside the object outline
    confidence: float = 0.0
    objects: list = field(default_factory=list)
    crop: tuple | None = None
    hit: Detection | None = None


def gaze_crop(gx: float, gy: float, w: int, h: int, frac: float):
    side = int(frac * min(w, h))
    side = max(side, min(w, h, 320))
    x1 = int(np.clip(gx - side / 2, 0, max(0, w - side)))
    y1 = int(np.clip(gy - side / 2, 0, max(0, h - side)))
    return x1, y1, min(w, x1 + side), min(h, y1 + side)


def pick_gazed_object(pt, dets: list[Detection], tol_px: float):
    """Inside an outline -> the smallest such object (most specific, e.g. toy over table).
    Otherwise the nearest object within the gaze-accuracy tolerance."""
    inside, near = [], []
    for d in dets:
        dist = distance_to_detection(np.asarray(pt, dtype=np.float32), d)
        if dist <= 0.0:
            inside.append(d)
        elif dist <= tol_px:
            near.append((dist, d))
    if inside:
        best = min(inside, key=lambda d: d.area)
        return best, 0.0
    if near:
        dist, best = min(near, key=lambda x: x[0])
        return best, dist
    return None, None


class GazeAnalyzer:
    def __init__(self, backend, settings):
        self.backend = backend
        self.s = settings

    def process(self, frame: np.ndarray, t: float, gaze) -> GazeSample:
        if gaze is None:
            return GazeSample(t, NO_GAZE)
        gx, gy, movement = gaze
        h, w = frame.shape[:2]
        if not (-0.05 * w <= gx <= 1.05 * w and -0.05 * h <= gy <= 1.05 * h):
            return GazeSample(t, NO_OBJECT, (gx, gy), movement)  # looking outside the scene camera
        crop = gaze_crop(gx, gy, w, h, self.s.gaze_crop_frac) if self.s.gaze_crop else None
        if crop is not None and (crop[2] - crop[0]) * (crop[3] - crop[1]) < 0.85 * w * h:
            x1, y1, x2, y2 = crop
            dets = [d.shifted(x1, y1) for d in self.backend.detect_objects(frame[y1:y2, x1:x2], self.s.det_imgsz)]
        else:
            crop = None
            dets = self.backend.detect_objects(frame, self.s.det_imgsz)
        if not self.s.include_people:
            dets = [d for d in dets if d.label != PERSON_LABEL]
        tol = self.s.gaze_tolerance * w
        hit, dist = pick_gazed_object((gx, gy), dets, tol)
        if hit is None:
            return GazeSample(t, NO_OBJECT, (gx, gy), movement, None, 0.0, dets, crop)
        conf = hit.conf * (1.0 - 0.5 * (dist / tol if tol > 0 else 0.0))
        return GazeSample(t, hit.label, (gx, gy), movement, dist, float(conf), dets, crop, hit)


def draw_gaze(frame: np.ndarray, s: GazeSample, color_of) -> np.ndarray:
    img = frame.copy()
    th = max(2, img.shape[0] // 300)
    if s.crop is not None:
        x1, y1, x2, y2 = s.crop
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    for det in s.objects:
        col = color_of(det.label)
        is_hit = s.hit is det
        if det.polygon is not None and is_hit:
            overlay = img.copy()
            cv2.fillPoly(overlay, [det.polygon.astype(np.int32)], col)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        b = det.box.astype(int)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, th * (3 if is_hit else 1))
        _label(img, det.label, (b[0], b[1]), col)
    if s.gaze is not None:
        gx, gy = int(s.gaze[0]), int(s.gaze[1])
        r = max(10, img.shape[1] // 60)
        cv2.circle(img, (gx, gy), r, (0, 0, 255), th + 1)
        cv2.circle(img, (gx, gy), 3, (0, 0, 255), -1)
    return img
