"""Colour-based stand-in for the YOLO back-end, used with the synthetic test videos.

red blob -> "ball", blue -> "block", green -> "toy car", magenta bodies -> people,
yellow dots -> wrists, cyan dots -> elbows.
"""
from __future__ import annotations

import cv2
import numpy as np

from gazeplay.detectors import Detection, Person

COLOR_RANGES = {  # HSV ranges
    "ball": [((0, 120, 80), (8, 255, 255)), ((172, 120, 80), (180, 255, 255))],
    "block": [((105, 120, 80), (130, 255, 255))],
    "toy car": [((45, 120, 80), (75, 255, 255))],
}
MAGENTA = [((140, 120, 80), (160, 255, 255))]
YELLOW = [((22, 120, 120), (35, 255, 255))]
CYAN = [((85, 120, 80), (98, 255, 255))]


def _mask(hsv, ranges):
    m = None
    for lo, hi in ranges:
        part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        m = part if m is None else cv2.bitwise_or(m, part)
    return m


def _blobs(mask, min_area):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            out.append(((x, y, x + w, y + h), cent[i], (lab == i).astype(np.uint8)))
    return out


class MockBackend:
    name = "mock colour detector"

    def __init__(self):
        self.classes = []
        self.calls = {"objects": 0, "people": 0}

    def configure(self, settings):
        self.settings = settings

    def describe(self):
        return {"backend": self.name, "object_model": "colour thresholds", "pose_model": "colour blobs",
                "device": "cpu"}

    def prepare_objects(self, labels):
        self.classes = list(labels)

    def reset_tracking(self):
        pass

    def detect_objects(self, image, imgsz):
        self.calls["objects"] += 1
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        min_area = max(20, image.shape[0] * image.shape[1] // 20000)
        dets = []
        for label, ranges in COLOR_RANGES.items():
            if label not in self.classes:
                continue
            for box, _, m in _blobs(_mask(hsv, ranges), min_area):
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                poly = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                dets.append(Detection(label, 0.9, np.array(box, np.float32), poly))
        return dets

    def detect_people(self, image, imgsz):
        self.calls["people"] += 1
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bodies = _blobs(_mask(hsv, MAGENTA), 400)
        wrists = [c for _, c, _ in _blobs(_mask(hsv, YELLOW), 15)]
        elbows = [c for _, c, _ in _blobs(_mask(hsv, CYAN), 15)]
        people = []
        bodies.sort(key=lambda b: b[0][0])
        for idx, (box, cen, _) in enumerate(bodies):
            x1, y1, x2, y2 = box
            k = np.zeros((17, 2), np.float32)
            c = np.zeros(17, np.float32)
            k[5], k[6] = (x1, y1), (x2, y1)        # shoulders
            k[11], k[12] = (x1, y2), (x2, y2)      # hips
            c[[5, 6, 11, 12]] = 0.9
            w = x2 - x1
            near = [p for p in wrists if x1 - 2.5 * w < p[0] < x2 + 2.5 * w and y1 - w < p[1] < y2 + 2 * w]
            near_e = [p for p in elbows if x1 - 2.5 * w < p[0] < x2 + 2.5 * w and y1 - w < p[1] < y2 + 2 * w]
            near.sort(key=lambda p: p[0])
            near_e.sort(key=lambda p: p[0])
            for j, p in enumerate(near[:2]):
                k[9 + j] = p
                c[9 + j] = 0.9
            for j, p in enumerate(near_e[:2]):
                k[7 + j] = p
                c[7 + j] = 0.9
            # head above torso
            bx = np.array([x1 - 0.3 * w, y1 - 1.0 * w, x2 + 0.3 * w, y2 + 1.2 * w], np.float32)
            people.append(Person(idx + 1, 0.9, bx, k, c))
        return people

    def discover_objects(self, images, imgsz=640):
        n = len(images)
        return {
            "ball": {"frames": n, "mean_conf": 0.8, "median_area": 0.01},
            "toy block": {"frames": int(0.9 * n), "mean_conf": 0.6, "median_area": 0.01},
            "car": {"frames": int(0.8 * n), "mean_conf": 0.55, "median_area": 0.01},
            "floor": {"frames": n, "mean_conf": 0.9, "median_area": 0.6},
            "person": {"frames": n, "mean_conf": 0.9, "median_area": 0.1},
            "table": {"frames": 3, "mean_conf": 0.4, "median_area": 0.2},
            "spoon": {"frames": 1, "mean_conf": 0.31, "median_area": 0.002},
        }
