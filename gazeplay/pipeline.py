"""End-to-end analysis: files in -> Excel / CSV / review video out."""
from __future__ import annotations

import math
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (DEFAULT_TOYS, DISCOVERY_STOPLIST, DISCOVERY_SYNONYMS, NO_GAZE, PERSON_LABEL,
                     Settings)
from .gaze_analysis import GazeAnalyzer, GazeSample, draw_gaze
from .gaze_data import load_gaze
from .report import summary_rows, write_csv, write_excel, write_json
from .review_video import FrameStore, compose
from .room_analysis import CHILD_NOT_VISIBLE, RoomAnalyzer, RoomSample, draw_room
from .sync import resolve_sync
from .timeline import (COL_GAZE, COL_INTER, COL_MATCH, COL_T, build_timeline, color_palette,
                       hex_to_bgr, per_object_stats, play_bouts, segments, summary_metrics)
from .video_io import SampledReader, probe_video

STAGES = [
    ("load", "Reading files", 0.02),
    ("sync", "Synchronising the two videos", 0.03),
    ("objects", "Identifying objects in the room", 0.07),
    ("room", "Room video: what is the child playing with?", 0.46),
    ("gaze", "Eye-tracking video: what is the child looking at?", 0.37),
    ("report", "Combining results and exporting", 0.05),
]
STAGE_INDEX = {k: i for i, (k, _, _) in enumerate(STAGES)}


class Cancelled(Exception):
    pass


class Progress:
    def __init__(self, callback=None, cancel: threading.Event | None = None):
        self.callback = callback or (lambda **kw: None)
        self.cancel = cancel
        self.stage = None
        self.frac = 0.0
        self.eta = None
        self.message = ""
        self.logs: list[str] = []
        self._last_emit = 0.0
        self.t0 = time.time()

    def overall(self) -> float:
        if self.stage is None:
            return 0.0
        i = STAGE_INDEX[self.stage]
        done = sum(w for _, _, w in STAGES[:i])
        return min(1.0, done + STAGES[i][2] * self.frac)

    def emit(self, force=False):
        now = time.time()
        if force or now - self._last_emit > 0.3:
            self._last_emit = now
            self.callback(stage=self.stage, stage_frac=self.frac, overall=self.overall(),
                          message=self.message, eta_s=self.eta, logs=list(self.logs))

    def start(self, stage: str, message: str = ""):
        self.check()
        self.stage, self.frac, self.message = stage, 0.0, message or dict((k, m) for k, m, _ in STAGES)[stage]
        self.emit(force=True)

    def update(self, frac: float, message: str | None = None, eta: float | None = None):
        self.check()
        self.frac = float(min(1.0, max(0.0, frac)))
        if message is not None:
            self.message = message
        self.eta = eta
        self.emit()

    def log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {msg}")
        self.emit(force=True)

    def check(self):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled("Analysis cancelled.")


def choose_discovered(stats: dict, n_frames: int, max_objects: int = 15):
    merged: dict[str, dict] = {}
    rows = []
    for raw, st in sorted(stats.items(), key=lambda kv: -kv[1]["frames"]):
        lab = DISCOVERY_SYNONYMS.get(raw, raw)
        reason = ""
        if lab in DISCOVERY_STOPLIST or raw in DISCOVERY_STOPLIST:
            reason = "background / person / too generic"
        elif st["median_area"] > 0.25:
            reason = "too large (furniture or background)"
        rows.append({"Detected Tag": raw, "Mapped To": lab, "Frames (%)": round(100 * st["frames"] / max(1, n_frames), 1),
                     "Mean Confidence": round(st["mean_conf"], 2), "_reason": reason})
        if not reason:
            m = merged.setdefault(lab, {"frames": 0, "conf": 0.0})
            m["frames"] = max(m["frames"], st["frames"])
            m["conf"] = max(m["conf"], st["mean_conf"])
    min_frames = max(2, math.ceil(0.08 * n_frames))
    ranked = sorted(merged.items(), key=lambda kv: (-kv[1]["frames"], -kv[1]["conf"]))
    selected = [lab for lab, m in ranked if m["frames"] >= min_frames][:max_objects]
    for r in rows:
        if r["_reason"]:
            r["Used?"] = "No - " + r["_reason"]
        elif r["Mapped To"] in selected:
            r["Used?"] = "Yes"
        else:
            r["Used?"] = "No - seen too rarely" if merged.get(r["Mapped To"], {}).get("frames", 0) < min_frames else "No - list limit"
        del r["_reason"]
    return selected, pd.DataFrame(rows)


def run_analysis(ego_video: str, room_video: str, gaze_file: str, out_dir: str | Path,
                 settings: Settings | None = None, progress_cb=None,
                 cancel: threading.Event | None = None, backend=None) -> dict:
    s = settings or Settings.from_preset("balanced")
    prog = Progress(progress_cb, cancel)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    timing: dict[str, float] = {}
    tmp = Path(tempfile.mkdtemp(prefix="gazeplay_"))
    t_all = time.time()

    try:
        # 1 ---------------------------------------------------------------- load
        prog.start("load")
        t_stage = time.time()
        ego_info = probe_video(ego_video)
        room_info = probe_video(room_video)
        prog.log(f"Eye-tracking video: {ego_info.width}x{ego_info.height}, {ego_info.fps:.2f} fps, "
                 f"{ego_info.duration:.1f} s")
        prog.log(f"Room video: {room_info.width}x{room_info.height}, {room_info.fps:.2f} fps, "
                 f"{room_info.duration:.1f} s")
        gaze = load_gaze(gaze_file, ego_info.width, ego_info.height, ego_info.duration,
                         fixations_only=s.fixations_only, time_offset_s=s.gaze_offset_s)
        warnings += gaze.info.get("warnings", [])
        prog.log(f"Gaze data: {gaze.info['valid_samples']} valid samples at ~{gaze.info['sampling_rate_hz']} Hz, "
                 f"time unit {gaze.info['time_unit']}, coordinates {gaze.info['coordinates']}")
        if gaze.end > ego_info.duration + 5:
            warnings.append(f"The gaze data ({gaze.end:.0f} s) is longer than the eye-tracking video "
                            f"({ego_info.duration:.0f} s). Check that the files belong together.")
        timing["load"] = time.time() - t_stage

        # 2 ---------------------------------------------------------------- sync
        prog.start("sync")
        t_stage = time.time()
        sync = resolve_sync(s, ego_video, room_video, prog.log)
        if sync.method != "audio" or sync.confidence != "low":
            prog.log(sync.message)
        if sync.confidence == "low" or (s.sync_mode == "audio" and sync.method == "none"):
            warnings.append(sync.message)
        offset = sync.offset_s
        timing["sync"] = time.time() - t_stage

        dt = 1.0 / s.sample_rate
        start = max(0.0, -offset, gaze.start)
        end = min(ego_info.duration, room_info.duration - offset, gaze.end)
        if s.max_duration_s:
            end = min(end, start + float(s.max_duration_s))
        if end - start < 2 * dt:
            raise ValueError(
                f"After synchronisation the videos and gaze data do not overlap in time "
                f"(offset {offset:+.2f} s). Check the files or set the offset manually.")
        times = list(np.round(np.arange(start, end - 1e-6, dt), 4))
        n = len(times)
        prog.log(f"Analysing {end - start:.1f} s at {s.sample_rate:g} samples/s ({n} time points).")

        # 3 ---------------------------------------------------------------- objects
        prog.start("objects")
        t_stage = time.time()
        if backend is None:
            from .detectors import UltralyticsBackend
            prog.update(0.05, "Loading detection models (first run downloads them)...")
            backend = UltralyticsBackend(log=prog.log)
        if hasattr(backend, "log"):
            backend.log = prog.log
        backend.configure(s)
        objects_found = None
        if s.objects:
            vocab, source = list(s.objects), "your list"
            prog.log("Objects to track (your list): " + ", ".join(vocab))
        else:
            prog.update(0.15, "Scanning both videos to find the objects in the room...")
            frames = []
            for path, info, shift in ((room_video, room_info, offset), (ego_video, ego_info, 0.0)):
                rd = SampledReader(path, info)
                for t in np.linspace(start + 1, max(start + 1, end - 1), 16):
                    fr = rd.read_at(float(t + shift))
                    if fr is not None:
                        frames.append(fr)
                rd.close()
            stats = backend.discover_objects(frames, s.det_imgsz)
            vocab, objects_found = choose_discovered(stats, len(frames))
            source = "found automatically"
            if not vocab:
                vocab, source = list(DEFAULT_TOYS), "default toy list"
                warnings.append("Automatic object discovery found no clear play objects; a default toy "
                                "list was used. For best results, type the toys used in the session.")
            prog.log(f"Objects to track ({source}): " + ", ".join(vocab))
        model_vocab = list(vocab)
        if s.include_people and PERSON_LABEL not in model_vocab:
            model_vocab.append(PERSON_LABEL)
        prog.update(0.7, "Preparing the object detector for: " + ", ".join(vocab))
        backend.prepare_objects(model_vocab)
        palette = color_palette(model_vocab)
        color_bgr = lambda lab: hex_to_bgr(palette.get(lab, "#888888"))  # noqa: E731
        timing["objects"] = time.time() - t_stage

        # 4 ---------------------------------------------------------------- room video
        prog.start("room")
        t_stage = time.time()
        backend.reset_tracking()
        room_an = RoomAnalyzer(backend, s)
        room_reader = SampledReader(room_video, room_info)
        room_store = FrameStore(tmp / "room.avi", s.sample_rate, room_info.width, room_info.height,
                                "Room camera") if s.make_review_video else None
        room_samples: list[RoomSample] = []
        for k, t in enumerate(times):
            fr = room_reader.read_at(t + offset)
            if fr is None:
                smp = RoomSample(t + offset, CHILD_NOT_VISIBLE)
            else:
                smp = room_an.process(fr, t + offset)
            room_samples.append(smp)
            if room_store:
                room_store.add(None if fr is None else draw_room(fr, smp, color_bgr), "room video ended")
            el = time.time() - t_stage
            rate = el / (k + 1)
            eta = rate * (n - k - 1) + rate * 0.7 * n
            prog.update((k + 1) / n, f"Room video: {k + 1}/{n} time points", eta)
        room_reader.close()
        if room_store:
            room_store.close()
        timing["room"] = time.time() - t_stage

        # 5 ---------------------------------------------------------------- eye-tracking video
        prog.start("gaze")
        t_stage = time.time()
        gaze_an = GazeAnalyzer(backend, s)
        ego_reader = SampledReader(ego_video, ego_info)
        ego_store = FrameStore(tmp / "ego.avi", s.sample_rate, ego_info.width, ego_info.height,
                               "Eye-tracking camera") if s.make_review_video else None
        gaze_samples: list[GazeSample] = []
        for k, t in enumerate(times):
            g = gaze.at(t, s.gaze_window_s)
            need = g is not None or ego_store is not None
            fr = ego_reader.read_at(t, need_frame=need)
            if g is None or fr is None:
                smp = GazeSample(t, NO_GAZE, None if g is None else (g[0], g[1]), None if g is None else g[2])
            else:
                smp = gaze_an.process(fr, t, g)
            gaze_samples.append(smp)
            if ego_store:
                ego_store.add(None if fr is None else draw_gaze(fr, smp, color_bgr), "video ended")
            el = time.time() - t_stage
            rate = el / (k + 1)
            prog.update((k + 1) / n, f"Eye-tracking video: {k + 1}/{n} time points",
                        rate * (n - k - 1) + 0.02 * n)
        ego_reader.close()
        if ego_store:
            ego_store.close()
        timing["gaze"] = time.time() - t_stage

        # 6 ---------------------------------------------------------------- combine & export
        prog.start("report")
        t_stage = time.time()
        timeline = build_timeline(times, room_samples, gaze_samples, offset, s)
        bouts = play_bouts(timeline, dt)
        per_obj = per_object_stats(timeline, bouts, dt, vocab, room_samples, gaze_samples)
        metrics = summary_metrics(timeline, bouts, dt)

        base = _safe_stem(Path(ego_video).stem) or "session"
        files = {}
        prog.update(0.2, "Writing Excel workbook...")
        context = _context_sections(s, ego_video, room_video, gaze_file, ego_info, room_info, gaze,
                                    sync, vocab, source, backend, warnings, timing, time.time() - t_all)
        files["excel"] = str(write_excel(out_dir / f"{base}_gaze_play_results.xlsx", timeline, bouts,
                                         per_obj, objects_found, metrics, context))
        files["csv"] = str(write_csv(out_dir / f"{base}_timeline.csv", timeline))
        if s.make_review_video:
            prog.update(0.5, "Rendering the side-by-side review video...")
            vid = compose(room_store, ego_store, out_dir / f"{base}_review.mp4", timeline, palette,
                          s.sample_rate, prog.log)
            files["video"] = str(vid) if vid else None
        timing["report"] = time.time() - t_stage
        timing["total"] = time.time() - t_all

        t_list = timeline[COL_T].tolist()
        result = {
            "metrics": metrics,
            "summary_rows": summary_rows(metrics),
            "sync": sync.to_dict(),
            "objects": vocab,
            "object_source": source,
            "colors": palette,
            "per_object": per_obj.to_dict("records"),
            "bouts": bouts.head(1000).to_dict("records"),
            "segments": {
                "interaction": segments(timeline[COL_INTER].tolist(), t_list, dt),
                "gaze": segments(timeline[COL_GAZE].tolist(), t_list, dt),
                "match": segments(timeline[COL_MATCH].tolist(), t_list, dt),
            },
            "t_start": float(times[0]),
            "t_end": float(times[-1] + dt),
            "dt": dt,
            "files": files,
            "warnings": warnings,
            "timing": {k: round(v, 1) for k, v in timing.items()},
            "inputs": {"ego": ego_info.to_dict(), "room": room_info.to_dict(), "gaze": {
                k: v for k, v in gaze.info.items() if k != "warnings"}},
            "backend": backend.describe() if hasattr(backend, "describe") else {},
        }
        result = clean_json(result)
        files["json"] = str(write_json(out_dir / f"{base}_summary.json", result))
        result["files"] = files
        prog.update(1.0, "Done")
        prog.log(f"Finished in {timing['total'] / 60:.1f} min. Results saved to {out_dir}")
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def clean_json(obj):
    """NaN/inf -> None and numpy scalars -> Python, so browsers can parse the JSON."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _safe_stem(stem: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in stem).strip()
    return keep[:60]


def _context_sections(s, ego_video, room_video, gaze_file, ego_info, room_info, gaze, sync, vocab,
                      source, backend, warnings, timing, elapsed):
    gi = gaze.info
    desc = backend.describe() if hasattr(backend, "describe") else {}
    return {
        "Synchronisation": [
            ("Method", {"audio": "Automatic (sound tracks)", "manual": "Manual offset",
                        "none": "None (offset 0)"}.get(sync.method, sync.method)),
            ("Offset (room time - eye-tracking time)", f"{sync.offset_s:+.3f} s"),
            ("Confidence", sync.confidence),
            ("Details", sync.message),
        ],
        "Input files": [
            ("Eye-tracking video", f"{Path(ego_video).name}  ({ego_info.width}x{ego_info.height}, "
                                   f"{ego_info.fps:.2f} fps, {ego_info.duration:.1f} s)"),
            ("Room video", f"{Path(room_video).name}  ({room_info.width}x{room_info.height}, "
                           f"{room_info.fps:.2f} fps, {room_info.duration:.1f} s)"),
            ("Gaze file", f"{Path(gaze_file).name}  ({gi['valid_samples']} valid samples, "
                          f"~{gi['sampling_rate_hz']} Hz)"),
            ("Gaze columns used", f"time: {gi['time_column']} ({gi['time_unit']}); x: {gi['x_column']}; "
                                  f"y: {gi['y_column']}"),
            ("Gaze coordinates", gi["coordinates"]),
        ],
        "Method and settings": [
            ("Objects tracked", f"{', '.join(vocab)}  ({source})"),
            ("Quality preset", f"{s.preset}  ({s.sample_rate:g} samples per second)"),
            ("Models", f"objects: {desc.get('object_model', '-')}; people/pose: {desc.get('pose_model', '-')}; "
                       f"device: {desc.get('device', '-')}"),
            ("Interaction rule", f"an object counts as 'played with' when a hand is within "
                                 f"{s.contact_distance:.2f} torso lengths of its outline"),
            ("Gaze rule", f"the object whose outline contains the gaze point, or the nearest one within "
                          f"{100 * s.gaze_tolerance:.1f}% of the frame width"),
            ("Smoothing", f"interaction {s.smooth_interaction_s:g} s, gaze {s.smooth_gaze_s:g} s (majority vote)"),
            ("Child selection", {"smallest": "smallest person in view", "largest": "largest person in view",
                                 "only": "most confident person"}.get(s.child_selection, s.child_selection)),
            ("People as gaze targets", "yes" if s.include_people else "no"),
            ("Fixations only", "yes" if gi.get("fixations_only") else "no"),
            ("Processing time", f"{elapsed / 60:.1f} min"),
        ],
        "Warnings": [(f"Warning {i + 1}", w) for i, w in enumerate(warnings)],
    }
