"""Detection back-end built on Ultralytics YOLOE (objects) and YOLO pose (people).

The rest of the pipeline only talks to the small interface defined here
(`prepare_objects`, `detect_objects`, `detect_people`, `discover_objects`,
`reset_tracking`), so another detector can be plugged in without touching
the analysis code (the test-suite uses a colour-based mock).
"""
from __future__ import annotations

import contextlib
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import MODELS_DIR


@dataclass
class Detection:
    label: str
    conf: float
    box: np.ndarray                    # [x1, y1, x2, y2] in frame pixels
    polygon: np.ndarray | None = None  # (N, 2) segmentation outline in frame pixels

    @property
    def area(self) -> float:
        return float(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))

    def shifted(self, dx: float, dy: float) -> "Detection":
        off = np.array([dx, dy, dx, dy], dtype=np.float32)
        poly = None if self.polygon is None else self.polygon + np.array([dx, dy], dtype=np.float32)
        return Detection(self.label, self.conf, self.box + off, poly)


@dataclass
class Person:
    track_id: int
    conf: float
    box: np.ndarray        # [x1, y1, x2, y2]
    kpts: np.ndarray       # (17, 2) COCO keypoints
    kconf: np.ndarray      # (17,)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def suppress_cross_class(dets: list[Detection], iou_thr: float = 0.7) -> list[Detection]:
    """One label per physical object: drop a box that overlaps a more confident one."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(box_iou(d.box, k.box) < iou_thr for k in kept):
            kept.append(d)
    return kept


@contextlib.contextmanager
def in_models_dir():
    """Ultralytics downloads weights / text encoders into the working directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prev = os.getcwd()
    os.chdir(MODELS_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


class UltralyticsBackend:
    name = "Ultralytics YOLOE + YOLO pose"

    def __init__(self, log=print):
        try:
            import ultralytics
            from ultralytics import YOLO, YOLOE
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "The 'ultralytics' package (8.4 or newer) is not installed. "
                "Run the setup script (setup_and_run.bat / setup_and_run.sh) first.") from exc
        import torch

        self._YOLO, self._YOLOE = YOLO, YOLOE
        self.version = ultralytics.__version__
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.log = log
        self.settings = None
        self.obj_model = None
        self.obj_model_name = None
        self.pose_model = None
        self.pose_model_name = None
        self.classes: list[str] | None = None
        self._tracking_ok = True

    # -- setup --------------------------------------------------------------
    def configure(self, settings):
        changed_obj = self.settings is None or settings.object_models != self.settings.object_models
        changed_pose = self.settings is None or settings.pose_models != self.settings.pose_models
        self.settings = settings
        if changed_obj:
            self.obj_model, self.classes = None, None
        if changed_pose:
            self.pose_model = None

    def describe(self) -> dict:
        return {"backend": self.name, "ultralytics": self.version, "device": self.device,
                "object_model": self.obj_model_name, "pose_model": self.pose_model_name}

    def _load_first(self, cls, names, kind):
        errors = []
        with in_models_dir():
            for n in names:
                try:
                    model = cls(n)
                    self.log(f"Loaded {kind} model {n}")
                    return model, n
                except Exception as exc:  # try the next candidate
                    errors.append(f"{n}: {exc}")
        raise RuntimeError(
            f"Could not load a {kind} model ({'; '.join(errors)}). The first run needs an internet "
            f"connection to download the models (they are then cached in the 'models' folder).")

    def prepare_objects(self, labels: list[str]):
        if self.obj_model is None:
            self.obj_model, self.obj_model_name = self._load_first(
                self._YOLOE, self.settings.object_models, "object detection")
        labels = [lab for lab in labels if lab.strip()]
        if labels == self.classes:
            return
        with in_models_dir():
            try:
                self.obj_model.set_classes(labels, self.obj_model.get_text_pe(labels))
            except TypeError:
                self.obj_model.set_classes(labels)
        self.classes = list(labels)

    def reset_tracking(self):
        # A fresh pose model gives fresh track IDs (cheap: the weights are cached on disk).
        self.pose_model = None

    # -- inference -----------------------------------------------------------
    @staticmethod
    def _names(res):
        names = res.names
        return (lambda i: names.get(int(i), str(i))) if isinstance(names, dict) else (lambda i: names[int(i)])

    def detect_objects(self, image: np.ndarray, imgsz: int) -> list[Detection]:
        res = self.obj_model.predict(image, imgsz=imgsz, conf=self.settings.det_conf,
                                     verbose=False, device=self.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        name_of = self._names(res)
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        polys = res.masks.xy if getattr(res, "masks", None) is not None else None
        dets = []
        for i in range(len(xyxy)):
            poly = None
            if polys is not None and i < len(polys) and len(polys[i]) >= 3:
                poly = np.asarray(polys[i], dtype=np.float32)
            dets.append(Detection(name_of(cls[i]), float(conf[i]), xyxy[i], poly))
        return suppress_cross_class(dets)

    def detect_people(self, image: np.ndarray, imgsz: int) -> list[Person]:
        if self.pose_model is None:
            self.pose_model, self.pose_model_name = self._load_first(
                self._YOLO, self.settings.pose_models, "pose")
            self._tracking_ok = True
        kwargs = dict(imgsz=imgsz, conf=self.settings.pose_conf, verbose=False, device=self.device)
        res = None
        if self._tracking_ok:
            try:
                res = self.pose_model.track(image, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
            except Exception as exc:
                self._tracking_ok = False
                self.log(f"Person tracking unavailable ({exc}); continuing frame-by-frame.")
        if res is None:
            res = self.pose_model.predict(image, **kwargs)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = res.boxes.conf.cpu().numpy()
        ids = (res.boxes.id.cpu().numpy().astype(int) if getattr(res.boxes, "id", None) is not None
               else -(np.arange(len(boxes)) + 1))
        xy = res.keypoints.xy.cpu().numpy()
        kc = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
              else np.ones(xy.shape[:2], dtype=np.float32))
        return [Person(int(ids[i]), float(confs[i]), boxes[i], xy[i], kc[i]) for i in range(len(boxes))]

    def discover_objects(self, images: list[np.ndarray], imgsz: int = 640) -> dict:
        model, name = self._load_first(self._YOLOE, self.settings.discovery_models,
                                       "prompt-free discovery")
        stats = defaultdict(lambda: {"frames": 0, "conf": [], "area": []})
        for img in images:
            res = model.predict(img, imgsz=imgsz, conf=0.30, verbose=False, device=self.device)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            name_of = self._names(res)
            h, w = img.shape[:2]
            seen = set()
            for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy()):
                label = str(name_of(k)).strip().lower()
                seen.add(label)
                stats[label]["conf"].append(float(c))
                stats[label]["area"].append(float((b[2] - b[0]) * (b[3] - b[1]) / (w * h)))
            for label in seen:
                stats[label]["frames"] += 1
        del model
        return {k: {"frames": v["frames"], "mean_conf": float(np.mean(v["conf"])),
                    "median_area": float(np.median(v["area"]))} for k, v in stats.items()}
