"""Room (external) video: find the child, their hands, and the object in their hands."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import (CHILD_NOT_VISIBLE, HANDS_NOT_VISIBLE, MULTI_SEP, NO_INTERACTION,
                     PERSON_LABEL)
from .detectors import Detection, Person

# COCO keypoint indices
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12
KP_OK = 0.30
WRIST_OK = 0.25


@dataclass
class RoomSample:
    t_room: float
    label: str
    objects_in_hand: tuple = ()
    confidence: float = 0.0
    child: Person | None = None
    hands: list = field(default_factory=list)
    scale: float = 0.0
    objects: list = field(default_factory=list)
    crop: tuple | None = None
    contacts: list = field(default_factory=list)   # (label, normalised distance, score)


def body_scale(p: Person) -> float:
    """Torso length in pixels (shoulders -> hips), a size reference that survives sitting."""
    k, c = p.kpts, p.kconf

    def ok(*idx):
        return all(c[i] >= KP_OK for i in idx)

    if ok(L_SHO, R_SHO, L_HIP, R_HIP):
        s = np.linalg.norm((k[L_SHO] + k[R_SHO]) / 2 - (k[L_HIP] + k[R_HIP]) / 2)
    elif ok(L_SHO, L_HIP):
        s = np.linalg.norm(k[L_SHO] - k[L_HIP])
    elif ok(R_SHO, R_HIP):
        s = np.linalg.norm(k[R_SHO] - k[R_HIP])
    elif ok(L_SHO, R_SHO):
        s = 1.2 * np.linalg.norm(k[L_SHO] - k[R_SHO])
    else:
        s = 0.30 * (p.box[3] - p.box[1])
    return float(max(s, 8.0))


def hand_points(p: Person) -> list[np.ndarray]:
    """Approximate hand centres: a bit beyond the wrist along the forearm."""
    pts = []
    for w, e in ((L_WRI, L_ELB), (R_WRI, R_ELB)):
        if p.kconf[w] >= WRIST_OK:
            pt = p.kpts[w].astype(np.float32)
            if p.kconf[e] >= WRIST_OK:
                pt = pt + 0.30 * (pt - p.kpts[e])
            pts.append(pt)
    return pts


def distance_to_detection(pt: np.ndarray, det: Detection) -> float:
    if det.polygon is not None and len(det.polygon) >= 3:
        d = cv2.pointPolygonTest(det.polygon.reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), True)
        return float(max(0.0, -d))
    x1, y1, x2, y2 = det.box
    dx = max(x1 - pt[0], 0.0, pt[0] - x2)
    dy = max(y1 - pt[1], 0.0, pt[1] - y2)
    return float(np.hypot(dx, dy))


class ChildSelector:
    """Pick the child among detected people, consistently over time.

    'smallest': the person with the smallest typical torso length (robust to
    sitting/standing); 'largest': the opposite; 'only': most confident person.
    A track keeps the role unless another person is clearly a better match.
    """

    def __init__(self, mode: str = "smallest"):
        self.mode = mode
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self.current_id: int | None = None

    def select(self, people: list[Person]) -> Person | None:
        if not people:
            return None
        for p in people:
            if p.track_id >= 0:
                self.history[p.track_id].append(body_scale(p))
        if self.mode == "only" or len(people) == 1:
            best = max(people, key=lambda p: p.conf)
            self.current_id = best.track_id
            return best

        def typical(p):
            h = self.history.get(p.track_id) if p.track_id >= 0 else None
            return float(np.median(h)) if h else body_scale(p)

        sign = 1.0 if self.mode != "largest" else -1.0
        ranked = sorted(people, key=lambda p: sign * typical(p))
        best = ranked[0]
        current = next((p for p in people if p.track_id == self.current_id and p.track_id >= 0), None)
        if current is not None and current is not best:
            # hysteresis: switch only when the other person is >20 % smaller (or larger)
            if sign * typical(current) <= sign * typical(best) * (1.2 if sign > 0 else 0.8):
                best = current
        self.current_id = best.track_id
        return best


def detection_crop(child: Person, hands, w: int, h: int, min_side: int = 320):
    """Region around the child (arm's reach + floor in front) for zoomed object detection."""
    x1, y1, x2, y2 = child.box
    bw, bh = x2 - x1, y2 - y1
    xs = [x1 - 0.5 * bw, x2 + 0.5 * bw]
    ys = [y1 - 0.25 * bh, y2 + 0.35 * bh]
    for hp in hands:
        xs += [hp[0] - 0.3 * bw, hp[0] + 0.3 * bw]
        ys += [hp[1] - 0.3 * bh, hp[1] + 0.3 * bh]
    cx1, cx2 = max(0, int(min(xs))), min(w, int(max(xs)))
    cy1, cy2 = max(0, int(min(ys))), min(h, int(max(ys)))
    side = max(min_side, int(0.35 * min(w, h)))
    if cx2 - cx1 < side:
        c = (cx1 + cx2) / 2
        cx1, cx2 = max(0, int(c - side / 2)), min(w, int(c + side / 2))
    if cy2 - cy1 < side:
        c = (cy1 + cy2) / 2
        cy1, cy2 = max(0, int(c - side / 2)), min(h, int(c + side / 2))
    if (cx2 - cx1) * (cy2 - cy1) > 0.7 * w * h:
        return None  # nearly the whole frame anyway
    return cx1, cy1, cx2, cy2


class RoomAnalyzer:
    def __init__(self, backend, settings):
        self.backend = backend
        self.s = settings
        self.selector = ChildSelector(settings.child_selection)

    def process(self, frame: np.ndarray, t_room: float) -> RoomSample:
        h, w = frame.shape[:2]
        people = self.backend.detect_people(frame, self.s.pose_imgsz)
        child = self.selector.select(people)
        if child is None:
            return RoomSample(t_room, CHILD_NOT_VISIBLE)
        hands = hand_points(child)
        scale = body_scale(child)
        if not hands:  # nothing to compare against - save the detector call
            return RoomSample(t_room, HANDS_NOT_VISIBLE, child=child, scale=scale)
        crop = detection_crop(child, hands, w, h) if self.s.room_crop else None
        if crop is not None:
            x1, y1, x2, y2 = crop
            dets = [d.shifted(x1, y1) for d in self.backend.detect_objects(frame[y1:y2, x1:x2], self.s.det_imgsz)]
        else:
            dets = self.backend.detect_objects(frame, self.s.det_imgsz)
        objects = [d for d in dets if d.label != PERSON_LABEL]
        sample = RoomSample(t_room, NO_INTERACTION, child=child, hands=hands, scale=scale,
                            objects=objects, crop=crop)

        best: dict[str, tuple[float, float]] = {}
        for det in objects:
            d = min(distance_to_detection(hp, det) for hp in hands) / scale
            if d <= self.s.contact_distance:
                score = det.conf * (1.0 - 0.5 * d / self.s.contact_distance)
                if det.label not in best or d < best[det.label][0]:
                    best[det.label] = (d, score)
        if best:
            ranked = sorted(best.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[:2]
            sample.contacts = [(lab, dist, sc) for lab, (dist, sc) in ranked]
            sample.objects_in_hand = tuple(sorted(lab for lab, _ in ranked))
            sample.label = MULTI_SEP.join(sample.objects_in_hand)
            sample.confidence = float(max(sc for _, (_, sc) in ranked))
        else:
            sample.confidence = float(min(1.0, child.conf))
        return sample


# ---------------------------------------------------------------------------
# Drawing (review video)
# ---------------------------------------------------------------------------
def draw_room(frame: np.ndarray, s: RoomSample, color_of) -> np.ndarray:
    img = frame.copy()
    th = max(2, img.shape[0] // 300)
    if s.crop is not None:
        x1, y1, x2, y2 = s.crop
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    for det in s.objects:
        col = color_of(det.label)
        held = det.label in s.objects_in_hand
        b = det.box.astype(int)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, th * (3 if held else 1))
        _label(img, det.label + (" (in hand)" if held else ""), (b[0], b[1]), col)
    if s.child is not None:
        b = s.child.box.astype(int)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), th)
        _label(img, "child", (b[0], b[1]), (90, 90, 90))
        for hp in s.hands:
            cv2.circle(img, (int(hp[0]), int(hp[1])), th * 5, (0, 215, 255), -1)
            cv2.circle(img, (int(hp[0]), int(hp[1])), th * 5, (0, 0, 0), max(1, th // 2))
    return img


def _label(img, text, org, color):
    scale = max(0.45, img.shape[0] / 1400)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(max(th + 6, org[1]))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
