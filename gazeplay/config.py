"""Settings, quality presets and object vocabularies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "GazePlay Analyzer"
APP_VERSION = "1.0.0"
APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models"
WORKSPACE_DIR = APP_DIR / "workspace"
RESULTS_DIR = APP_DIR / "results"

# ---------------------------------------------------------------------------
# Labels written to the output file
# ---------------------------------------------------------------------------
NO_INTERACTION = "No object in hand"     # child visible, hands free
CHILD_NOT_VISIBLE = "Child not visible"  # no child found in room video
HANDS_NOT_VISIBLE = "Hands not visible"  # child found, wrists occluded
NO_GAZE = "No gaze data"                 # blink / tracking loss / no sample
NO_OBJECT = "No object"                  # valid gaze, but not on a known object
OUTSIDE_VIEW = "Outside camera view"     # valid gaze outside the scene camera (e.g. looking down at the lap)
PERSON_LABEL = "person"
MULTI_SEP = " + "                        # joins several objects held at once

NON_OBJECT_INTERACTION = {NO_INTERACTION, CHILD_NOT_VISIBLE, HANDS_NOT_VISIBLE}
NON_OBJECT_GAZE = {NO_GAZE, NO_OBJECT, OUTSIDE_VIEW}

# Used only when the object list is left empty AND automatic discovery fails.
DEFAULT_TOYS = [
    "ball", "teddy bear", "doll", "toy car", "toy truck", "block",
    "stacking cup", "stacking rings", "puzzle", "book", "cup", "bowl",
    "spoon", "bottle", "toy phone", "rattle", "xylophone", "drum",
    "shape sorter", "stuffed toy", "toy animal", "crayon", "plate", "box",
]

# Tags from the prompt-free model that are never "play objects".
DISCOVERY_STOPLIST = {
    # people & body parts
    "person", "people", "man", "woman", "boy", "girl", "child", "kid", "baby",
    "toddler", "infant", "family", "mother", "father", "parent", "adult",
    "face", "human face", "hand", "arm", "leg", "foot", "finger", "head",
    "hair", "eye", "body", "skin", "smile", "sit", "play", "crawl",
    # clothing & worn items
    "shirt", "t-shirt", "pants", "jeans", "dress", "sock", "shoe", "clothing",
    "sweater", "jacket", "hat", "glasses", "eyeglasses", "sunglasses", "diaper",
    "headband", "eyewear", "goggles",
    # room structure & furniture
    "floor", "wall", "ceiling", "room", "window", "door", "carpet", "rug",
    "mat", "floor mat", "play mat", "curtain", "light", "lamp", "ceiling light",
    "lighting", "wood", "wood floor", "hardwood", "laminate", "tile", "home",
    "house", "interior", "living room", "bedroom", "playroom", "play room",
    "kindergarten", "nursery", "table", "desk", "chair", "couch", "sofa", "bed",
    "shelf", "bookshelf", "cabinet", "cupboard", "drawer", "stool", "bench",
    "crib", "high chair", "radiator", "outlet", "cable", "wire", "mirror",
    "plant", "houseplant", "flower pot", "picture frame", "poster", "painting",
    "camera", "tripod", "pillow", "cushion", "blanket",
    # too generic to separate two objects
    "toy", "toys", "object", "item", "thing", "plaything", "stuff", "game",
    "plastic", "fabric", "white", "black", "red", "blue", "green", "yellow",
    "pink", "orange", "purple", "color", "colorful", "pattern", "design",
    "logo", "text", "sign", "stripe", "circle", "square", "shape",
}

# Collapse near-synonyms so both camera views use the same name.
DISCOVERY_SYNONYMS = {
    "toy block": "block", "building block": "block", "cube": "block",
    "lego": "block", "wooden block": "block", "brick": "block",
    "stuffed animal": "stuffed toy", "plush": "stuffed toy",
    "soft toy": "stuffed toy", "plush toy": "stuffed toy",
    "teddy": "teddy bear",
    "toy ball": "ball", "football": "ball", "soccer ball": "ball",
    "sports ball": "ball", "beach ball": "ball", "basketball": "ball",
    "car": "toy car", "toy vehicle": "toy car", "model car": "toy car",
    "truck": "toy truck", "picture book": "book", "mug": "cup",
    "plastic cup": "cup", "phone": "toy phone", "mobile phone": "toy phone",
    "cellphone": "toy phone", "telephone": "toy phone",
    "jigsaw puzzle": "puzzle", "jigsaw": "puzzle",
}

# ---------------------------------------------------------------------------
# Quality presets (CPU friendly). Model names fall back automatically when the
# newer YOLO26-based checkpoints are unavailable in the installed ultralytics.
# ---------------------------------------------------------------------------
PRESETS = {
    "fast": dict(
        sample_rate=3.0, det_imgsz=512, pose_imgsz=512,
        object_models=["yoloe-26s-seg.pt", "yoloe-11s-seg.pt"],
        pose_models=["yolo26n-pose.pt", "yolo11n-pose.pt"],
    ),
    "balanced": dict(
        sample_rate=5.0, det_imgsz=640, pose_imgsz=640,
        object_models=["yoloe-26s-seg.pt", "yoloe-11s-seg.pt"],
        pose_models=["yolo26n-pose.pt", "yolo11n-pose.pt"],
    ),
    "accurate": dict(
        sample_rate=10.0, det_imgsz=800, pose_imgsz=800,
        object_models=["yoloe-26m-seg.pt", "yoloe-11m-seg.pt"],
        pose_models=["yolo26s-pose.pt", "yolo11s-pose.pt"],
    ),
}

PRESET_DESCRIPTIONS = {
    "fast": "3 samples/s - quickest, good for a first look",
    "balanced": "5 samples/s - recommended",
    "accurate": "10 samples/s, larger models - slowest",
}


@dataclass
class Settings:
    preset: str = "balanced"
    # Objects to look for. Empty list = discover automatically from the videos.
    objects: list[str] = field(default_factory=list)

    # Sampling / models (filled from the preset)
    sample_rate: float = 5.0
    det_imgsz: int = 640
    pose_imgsz: int = 640
    object_models: list[str] = field(default_factory=lambda: list(PRESETS["balanced"]["object_models"]))
    pose_models: list[str] = field(default_factory=lambda: list(PRESETS["balanced"]["pose_models"]))
    discovery_models: list[str] = field(default_factory=lambda: ["yoloe-26s-seg-pf.pt", "yoloe-11s-seg-pf.pt"])
    det_conf: float = 0.25
    pose_conf: float = 0.30

    # Synchronisation
    sync_mode: str = "audio"        # "audio" | "manual" | "none" | "segments"
    manual_offset_s: float = 0.0    # room-video time at which the eye-tracking video starts
    gaze_offset_s: float = 0.0      # added to gaze timestamps
    # Segments chosen in the sync window (sync_mode="segments"), each on its own video's clock.
    # The two start points mark the same moment, so offset = room_start_s - ego_start_s.
    ego_start_s: float | None = None
    ego_end_s: float | None = None
    room_start_s: float | None = None
    room_end_s: float | None = None
    sync_offset_s: float | None = None     # the offset the user confirmed (room = ego + offset)
    audio_offset_s: float | None = None    # what the automatic audio sync found (for the report)
    audio_confidence: str | None = None

    # Interaction (room video)
    child_selection: str = "only"       # "only" | "smallest" | "largest"
    contact_distance: float = 0.30      # hand-object distance, in child torso lengths
    room_crop: bool = True              # detect objects in a zoomed crop around the child

    # Gaze (eye-tracking video)
    gaze_tolerance: float = 0.02        # fraction of frame width (~1.5 deg of visual angle)
    gaze_window_s: float = 0.05         # +/- window for averaging gaze samples
    fixations_only: bool = False
    include_people: bool = False        # "person" can be a gaze target (e.g. parent's face)
    gaze_crop: bool = True              # detect objects in a zoomed crop around the gaze point
    gaze_crop_frac: float = 0.67        # crop side, fraction of the shorter frame side

    # Temporal smoothing
    smooth_interaction_s: float = 1.0
    smooth_gaze_s: float = 0.4

    # Output
    make_review_video: bool = True
    max_duration_s: float | None = None  # analyse only the first N seconds (testing)

    @classmethod
    def from_preset(cls, preset: str = "balanced", **overrides) -> "Settings":
        preset = preset if preset in PRESETS else "balanced"
        values = dict(PRESETS[preset])
        values["object_models"] = list(values["object_models"])
        values["pose_models"] = list(values["pose_models"])
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(preset=preset, **values)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_object_list(text: str | list | None) -> list[str]:
    """'Red ball, teddy bear; blocks' -> ['red ball', 'teddy bear', 'blocks'] (deduplicated)."""
    if not text:
        return []
    if isinstance(text, (list, tuple)):
        items = [str(t) for t in text]
    else:
        items = str(text).replace(";", ",").replace("\n", ",").split(",")
    seen, out = set(), []
    for item in items:
        name = " ".join(item.strip().lower().split())
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out
