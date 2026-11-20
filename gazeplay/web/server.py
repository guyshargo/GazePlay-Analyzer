"""Local web interface (runs on http://127.0.0.1 only; nothing leaves the computer)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from ..config import (APP_NAME, APP_VERSION, PRESET_DESCRIPTIONS, PRESETS, RESULTS_DIR, WORKSPACE_DIR,
                      Settings, parse_object_list)
from ..gaze_data import summarize_gaze_file
from ..pipeline import STAGES, Cancelled, run_analysis
from ..video_io import probe_video

HERE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))
app.config["MAX_CONTENT_LENGTH"] = None
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm", ".mts", ".mpg", ".mpeg"}
GAZE_EXT = {".tsv", ".csv", ".txt", ".xlsx", ".xls", ".xlsm"}
UPLOAD_DIR = WORKSPACE_DIR / "uploads"

UPLOADS: dict[str, dict] = {}
JOBS: dict[str, "Job"] = {}
_backend = None
_backend_lock = threading.Lock()
_run_lock = threading.Lock()


class Job:
    def __init__(self, job_id: str, out_dir: Path):
        self.id = job_id
        self.out_dir = out_dir
        self.state = "queued"
        self.progress = {"stage": None, "stage_frac": 0.0, "overall": 0.0, "message": "Waiting to start...",
                         "eta_s": None, "logs": []}
        self.result = None
        self.error = None
        self.cancel = threading.Event()
        self.started = time.time()

    def update(self, **kw):
        self.progress.update(kw)

    def to_dict(self):
        d = {"id": self.id, "state": self.state, "error": self.error,
             "elapsed_s": round(time.time() - self.started, 1), **self.progress}
        d["logs"] = d.get("logs", [])[-200:]
        if self.state == "done" and self.result:
            d["result"] = self.result
        return d


def _get_backend():
    global _backend
    with _backend_lock:
        if _backend is None:
            if os.environ.get("GAZEPLAY_BACKEND") == "mock":  # test mode, see tests/
                sys.path.insert(0, str(HERE.parent.parent / "tests"))
                from mock_backend import MockBackend
                _backend = MockBackend()
            else:
                from ..detectors import UltralyticsBackend
                _backend = UltralyticsBackend()
        return _backend


def _safe_name(name: str) -> str:
    name = Path(name).name
    keep = "".join(c if (c.isalnum() or c in "-_. ") else "_" for c in name).strip()
    return keep[:120] or "file"


def _fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


# ---------------------------------------------------------------------------
# Pages & config
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html", app_name=APP_NAME, version=APP_VERSION)


@app.get("/api/config")
def config():
    import importlib.util
    has_ultra = importlib.util.find_spec("ultralytics") is not None
    return jsonify({
        "app": APP_NAME, "version": APP_VERSION,
        "presets": [{"id": k, "label": k.capitalize(), "description": PRESET_DESCRIPTIONS[k],
                     "sample_rate": v["sample_rate"]} for k, v in PRESETS.items()],
        "defaults": Settings.from_preset("balanced").to_dict(),
        "stages": [{"id": k, "label": label} for k, label, _ in STAGES],
        "detector_ready": has_ultra or os.environ.get("GAZEPLAY_BACKEND") == "mock",
        "results_dir": str(RESULTS_DIR),
    })


# ---------------------------------------------------------------------------
# Uploads (raw streamed body, so multi-GB videos never sit in memory)
# ---------------------------------------------------------------------------
@app.post("/api/upload")
def upload():
    kind = request.args.get("kind", "")
    if kind not in ("ego", "room", "gaze"):
        return jsonify(error="Unknown file slot."), 400
    name = urllib.parse.unquote(request.headers.get("X-File-Name", "file"))
    ext = Path(name).suffix.lower()
    allowed = GAZE_EXT if kind == "gaze" else VIDEO_EXT
    if ext not in allowed:
        what = "a gaze data file (.tsv, .csv or .xlsx)" if kind == "gaze" else "a video file (.mp4, .mov, .avi...)"
        return jsonify(error=f"'{Path(name).name}' is not {what}."), 400
    uid = uuid.uuid4().hex[:12]
    folder = UPLOAD_DIR / uid
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / _safe_name(name)
    size = 0
    try:
        with open(dest, "wb") as fh:
            while True:
                chunk = request.stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                size += len(chunk)
        if size == 0:
            raise ValueError("The file is empty.")
        if kind == "gaze":
            info = summarize_gaze_file(dest)
            if not info.get("ok"):
                raise ValueError(info.get("error", "Unrecognised gaze file."))
            rate = f" at {info['sampling_rate_hz']:.0f} Hz" if info.get("sampling_rate_hz") else ""
            summary = (f"{info['rows']:,} rows{rate} | {_fmt_dur(info.get('duration_s', 0))} | "
                       f"{info.get('valid_pct', 0):.0f}% valid gaze")
        else:
            vi = probe_video(str(dest))
            info = vi.to_dict()
            summary = f"{_fmt_dur(vi.duration)} | {vi.width}x{vi.height} | {vi.fps:.0f} fps"
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify(error=str(exc)), 400
    UPLOADS[uid] = {"id": uid, "kind": kind, "name": Path(name).name, "path": str(dest), "size": size,
                    "info": info, "summary": summary}
    return jsonify({k: v for k, v in UPLOADS[uid].items() if k != "path"})


@app.delete("/api/upload/<uid>")
def delete_upload(uid):
    item = UPLOADS.pop(uid, None)
    if item:
        shutil.rmtree(Path(item["path"]).parent, ignore_errors=True)
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def _settings_from_request(data: dict) -> Settings:
    adv = data.get("advanced") or {}
    s = Settings.from_preset(data.get("preset", "balanced"))
    s.objects = parse_object_list(data.get("objects", ""))
    s.sync_mode = adv.get("sync_mode", "audio") if adv.get("sync_mode") in ("audio", "manual", "none") else "audio"
    s.manual_offset_s = float(adv.get("manual_offset_s") or 0.0)
    s.gaze_offset_s = float(adv.get("gaze_offset_s") or 0.0)
    if adv.get("child_selection") in ("smallest", "largest", "only"):
        s.child_selection = adv["child_selection"]
    s.contact_distance = {"strict": 0.20, "normal": 0.30, "lenient": 0.45}.get(adv.get("contact", "normal"), 0.30)
    if adv.get("gaze_tolerance_pct") not in (None, ""):
        s.gaze_tolerance = max(0.005, min(0.10, float(adv["gaze_tolerance_pct"]) / 100.0))
    s.include_people = bool(adv.get("include_people", True))
    s.fixations_only = bool(adv.get("fixations_only", False))
    s.make_review_video = bool(adv.get("review_video", True))
    if adv.get("max_duration_s"):
        s.max_duration_s = float(adv["max_duration_s"])
    return s


def _worker(job: Job, paths: dict, settings: Settings):
    with _run_lock:
        if job.cancel.is_set():
            job.state = "cancelled"
            return
        job.state = "running"
        job.started = time.time()
        try:
            job.update(message="Loading detection models...")
            backend = _get_backend()
            result = run_analysis(paths["ego"], paths["room"], paths["gaze"], job.out_dir, settings,
                                  progress_cb=job.update, cancel=job.cancel, backend=backend)
            result["files"] = {k: v for k, v in result["files"].items() if v}
            result["downloads"] = {k: f"/api/download/{job.id}/{k}" for k in result["files"]}
            result["out_dir"] = str(job.out_dir)
            result["file_names"] = {k: Path(v).name for k, v in result["files"].items()}
            job.result = result
            job.state = "done"
        except Cancelled:
            job.state = "cancelled"
            job.error = "Analysis cancelled."
        except Exception as exc:
            traceback.print_exc()
            job.state = "error"
            job.error = str(exc) or exc.__class__.__name__


@app.post("/api/run")
def run():
    data = request.get_json(force=True, silent=True) or {}
    paths = {}
    for kind in ("ego", "room", "gaze"):
        item = UPLOADS.get(data.get(kind, ""))
        if not item or not Path(item["path"]).exists():
            return jsonify(error=f"Please add the {'eye-tracking video' if kind == 'ego' else 'room video' if kind == 'room' else 'gaze data file'} first."), 400
        paths[kind] = item["path"]
    try:
        settings = _settings_from_request(data)
    except (TypeError, ValueError) as exc:
        return jsonify(error=f"Invalid setting: {exc}"), 400
    stem = Path(UPLOADS[data["ego"]]["name"]).stem
    out_dir = RESULTS_DIR / f"{_safe_name(stem)[:40]}_{datetime.now():%Y-%m-%d_%H%M%S}"
    job = Job(uuid.uuid4().hex[:12], out_dir)
    JOBS[job.id] = job
    threading.Thread(target=_worker, args=(job, paths, settings), daemon=True).start()
    return jsonify(job_id=job.id)


@app.get("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job."), 404
    return jsonify(job.to_dict())


@app.post("/api/cancel/<job_id>")
def cancel(job_id):
    job = JOBS.get(job_id)
    if job:
        job.cancel.set()
    return jsonify(ok=True)


@app.get("/api/download/<job_id>/<kind>")
def download(job_id, kind):
    job = JOBS.get(job_id)
    if not job or not job.result or kind not in job.result["files"]:
        abort(404)
    path = Path(job.result["files"][kind])
    if not path.exists():
        abort(404)
    inline = kind == "video" and request.args.get("inline") == "1"
    return send_file(path, as_attachment=not inline, download_name=path.name, conditional=True)


@app.post("/api/open-folder/<job_id>")
def open_folder(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    folder = str(job.out_dir)
    try:
        if platform.system() == "Windows":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
        return jsonify(ok=True)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc), path=folder)


def cleanup_old_uploads(max_age_h: float = 24):
    if not UPLOAD_DIR.exists():
        return
    cutoff = time.time() - max_age_h * 3600
    for d in UPLOAD_DIR.iterdir():
        try:
            if d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
