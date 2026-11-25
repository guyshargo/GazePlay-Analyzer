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

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from ..config import (APP_NAME, APP_VERSION, PRESET_DESCRIPTIONS, PRESETS, RESULTS_DIR, WORKSPACE_DIR,
                      Settings, parse_object_list)
from ..gaze_data import summarize_gaze_file
from ..pipeline import STAGES, Cancelled, run_analysis
from ..sync import AUDIO_SR, extract_audio, ffmpeg_exe, offset_from_envelopes, onset_envelope  # noqa: F401
from ..video_io import probe_video

HERE = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))
app.config["MAX_CONTENT_LENGTH"] = None
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm", ".mts", ".mpg", ".mpeg"}
GAZE_EXT = {".tsv", ".csv", ".txt", ".xlsx", ".xls", ".xlsm", ".gz", ".json", ".jsonl", ".ndjson"}
UPLOAD_DIR = WORKSPACE_DIR / "uploads"

UPLOADS: dict[str, dict] = {}
JOBS: dict[str, "Job"] = {}
_backend = None
_backend_lock = threading.Lock()
_run_lock = threading.Lock()
_AUDIO: dict[str, dict] = {}        # upload id -> {"env", "wave", "duration"} (sync window)
_THUMBS: dict[tuple, bytes] = {}
_PROXY: dict[str, dict] = {}        # upload id -> {"state", "progress", "path", "error"}
_SYNC_JOBS: dict[str, dict] = {}    # "ego_id|room_id" -> automatic audio-sync job (runs once per pair)
_media_locks: dict[str, threading.Lock] = {}


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
                from tests.mock_backend import MockBackend
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
        what = ("a gaze data file (Glasses 3 gazedata.gz, or a Pro Lab .tsv/.csv/.xlsx export)" if kind == "gaze"
                else "a video file (.mp4, .mov, .avi...)")
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
            source = "Glasses 3 | " if "Glasses 3" in info.get("format", "") else ""
            summary = (f"{source}{info['rows']:,} samples{rate} | {_fmt_dur(info.get('duration_s', 0))} | "
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
    _AUDIO.pop(uid, None)
    _PROXY.pop(uid, None)
    for key in [k for k in _SYNC_JOBS if uid in k.split("|")]:
        _SYNC_JOBS.pop(key, None)
    for key in [k for k in _THUMBS if k[0] == uid]:
        _THUMBS.pop(key, None)
    if item:
        shutil.rmtree(Path(item["path"]).parent, ignore_errors=True)
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Sync window: streaming, thumbnails, waveform, sound-based suggestion, preview proxy
# ---------------------------------------------------------------------------
def _upload_or_404(uid: str) -> dict:
    item = UPLOADS.get(uid)
    if not item or item["kind"] not in ("ego", "room") or not Path(item["path"]).exists():
        abort(404)
    return item


def _lock_for(uid: str) -> threading.Lock:
    with _backend_lock:
        return _media_locks.setdefault(uid, threading.Lock())


@app.get("/api/media/<uid>")
def media(uid):
    item = _upload_or_404(uid)
    path = item["path"]
    if request.args.get("proxy") == "1":
        p = _PROXY.get(uid, {})
        if p.get("state") != "done":
            abort(404)
        path = p["path"]
    # MP4-family containers (.mp4/.mov/.m4v) are sniffed fine by browsers as video/mp4
    mime = {".webm": "video/webm", ".ogv": "video/ogg"}.get(Path(path).suffix.lower(), "video/mp4")
    return send_file(path, mimetype=mime, conditional=True)


@app.get("/api/thumbs/<uid>")
def thumbs(uid):
    item = _upload_or_404(uid)
    n = max(4, min(40, int(request.args.get("n", 12))))
    key = (uid, n)
    if key not in _THUMBS:
        with _lock_for(uid):
            if key not in _THUMBS:
                h = 72
                cap = cv2.VideoCapture(item["path"])
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or int(item["info"].get("frames", 1))
                w = max(2, int(round(h * item["info"]["width"] / max(1, item["info"]["height"]))))
                tiles = []
                for i in range(n):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
                    ok, fr = cap.read()
                    tiles.append(cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA) if ok
                                 else np.zeros((h, w, 3), np.uint8))
                cap.release()
                ok, buf = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 72])
                _THUMBS[key] = buf.tobytes()
    return Response(_THUMBS[key], mimetype="image/jpeg", headers={"Cache-Control": "max-age=3600"})


def _audio_info(uid: str) -> dict:
    if uid not in _AUDIO:
        with _lock_for(uid):
            if uid not in _AUDIO:
                audio = extract_audio(UPLOADS[uid]["path"])
                if audio is None:
                    _AUDIO[uid] = {"env": None, "wave": [], "duration": 0.0}
                else:
                    bins = 1200
                    n = len(audio) // bins * bins
                    peaks = np.abs(audio[:n]).reshape(bins, -1).max(axis=1) if n else np.zeros(bins)
                    ref = float(np.percentile(peaks, 99)) or 1.0
                    wave = np.sqrt(np.clip(peaks / ref, 0, 1))  # sqrt: quiet sounds stay visible
                    _AUDIO[uid] = {"env": onset_envelope(audio), "wave": [round(float(v), 3) for v in wave],
                                   "duration": len(audio) / AUDIO_SR}
    return _AUDIO[uid]


@app.get("/api/waveform/<uid>")
def waveform(uid):
    _upload_or_404(uid)
    info = _audio_info(uid)
    return jsonify(has_audio=info["env"] is not None, peaks=info["wave"], duration=info["duration"])


def _run_sync_job(key: str, ego_id: str, room_id: str):
    """Automatic audio synchronisation, in the background: read both sound tracks (in parallel),
    then cross-correlate them. Started once per video pair; never re-run by manual adjustments."""
    job = _SYNC_JOBS[key]
    try:
        job.update(step="Reading the sound tracks of both videos...", progress=0.1)
        done = {}

        def read(uid):
            done[uid] = _audio_info(uid)
            job["progress"] = round(job["progress"] + 0.35, 2)

        threads = [threading.Thread(target=read, args=(u,), daemon=True) for u in (ego_id, room_id)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        a, b = done.get(ego_id), done.get(room_id)
        job.update(step="Comparing the two sound tracks...", progress=0.85)
        if not a or not b or a["env"] is None or b["env"] is None:
            which = "eye-tracking video" if (not a or a["env"] is None) else "room video"
            job.update(state="done", progress=1.0, available=False, offset_s=0.0, confidence="none",
                       message=f"The {which} has no usable sound track, so the videos could not be synchronised "
                               f"automatically. The offset starts at 0 s - set it by hand.")
            return
        offset, conf, z = offset_from_envelopes(a["env"], b["env"])
        job.update(state="done", progress=1.0, available=True, offset_s=round(float(offset), 3),
                   confidence=conf, score=round(float(z), 1), message="")
    except Exception as exc:  # report, never crash the server
        job.update(state="done", progress=1.0, available=False, offset_s=0.0, confidence="none",
                   message=f"Automatic synchronisation failed ({exc}). The offset starts at 0 s - set it by hand.")


@app.post("/api/sync/start")
def sync_start():
    data = request.get_json(force=True, silent=True) or {}
    ego, room = _upload_or_404(data.get("ego", "")), _upload_or_404(data.get("room", ""))
    key = f"{ego['id']}|{room['id']}"
    with _backend_lock:
        if key not in _SYNC_JOBS:
            _SYNC_JOBS[key] = {"id": key, "state": "running", "step": "Starting...", "progress": 0.0,
                               "started": time.time()}
            threading.Thread(target=_run_sync_job, args=(key, ego["id"], room["id"]), daemon=True).start()
    return jsonify(_SYNC_JOBS[key])


@app.get("/api/sync/status")
def sync_status():
    job = _SYNC_JOBS.get(request.args.get("key", ""))
    if not job:
        return jsonify(error="Unknown synchronisation job."), 404
    return jsonify(job)


def _make_proxy(uid: str):
    item = UPLOADS[uid]
    out = Path(item["path"]).parent / "preview_proxy.mp4"
    st = _PROXY[uid]
    exe = ffmpeg_exe()
    if not exe:
        st.update(state="error", error="ffmpeg is not available to convert this video.")
        return
    dur = float(item["info"].get("duration_s") or 1.0)
    cmd = [exe, "-y", "-v", "error", "-nostdin", "-i", item["path"], "-vf", "scale=-2:'min(540,ih)'",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-g", "25", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-progress", "pipe:1", str(out)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            if line.startswith("out_time_ms="):
                try:
                    st["progress"] = min(0.99, int(line.split("=")[1]) / 1e6 / dur)
                except ValueError:
                    pass
        proc.wait()
        if proc.returncode == 0 and out.exists():
            st.update(state="done", progress=1.0, path=str(out))
        else:
            st.update(state="error", error=(proc.stderr.read() or "conversion failed")[-300:])
    except Exception as exc:
        st.update(state="error", error=str(exc))


@app.post("/api/proxy/<uid>")
def start_proxy(uid):
    _upload_or_404(uid)
    with _lock_for(uid):
        if uid not in _PROXY or _PROXY[uid]["state"] == "error":
            _PROXY[uid] = {"state": "running", "progress": 0.0}
            threading.Thread(target=_make_proxy, args=(uid,), daemon=True).start()
    return jsonify(_PROXY[uid])


@app.get("/api/proxy/<uid>")
def proxy_status(uid):
    _upload_or_404(uid)
    st = _PROXY.get(uid, {"state": "none"})
    return jsonify({k: v for k, v in st.items() if k != "path"})


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def _settings_from_request(data: dict) -> Settings:
    adv = data.get("advanced") or {}
    s = Settings.from_preset(data.get("preset", "balanced"))
    s.objects = parse_object_list(data.get("objects", ""))
    apply_confirmed_sync(s, data.get("sync"), UPLOADS[data["ego"]]["info"], UPLOADS[data["room"]]["info"])
    s.gaze_offset_s = float(adv.get("gaze_offset_s") or 0.0)
    if adv.get("child_selection") in ("only", "smallest", "largest"):
        s.child_selection = adv["child_selection"]
    s.contact_distance = {"strict": 0.20, "normal": 0.30, "lenient": 0.45}.get(adv.get("contact", "normal"), 0.30)
    if adv.get("gaze_tolerance_pct") not in (None, ""):
        s.gaze_tolerance = max(0.005, min(0.10, float(adv["gaze_tolerance_pct"]) / 100.0))
    s.include_people = bool(adv.get("include_people", False))
    s.fixations_only = bool(adv.get("fixations_only", False))
    s.make_review_video = True          # the side-by-side review video is always produced
    return s


SYNC_TOLERANCE_S = 0.06   # float rounding / container duration differences


def apply_confirmed_sync(s: Settings, sync: dict | None, ego_info: dict, room_info: dict):
    """Use exactly the synchronisation the user confirmed: room time = eye-tracking time + offset,
    analysed part = [ego_start, ego_end] on the eye-tracking clock (and the same moments + offset
    on the room clock)."""
    if not sync or not sync.get("confirmed"):
        raise ValueError("Please confirm the video synchronisation first.")
    try:
        off = round(float(sync["offset"]), 6)
        e0, e1 = float(sync["ego_start"]), float(sync["ego_end"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("The confirmed synchronisation is incomplete - please confirm it again.")
    d_ego, d_room = float(ego_info.get("duration_s") or 0), float(room_info.get("duration_s") or 0)
    if not (-SYNC_TOLERANCE_S <= e0 < e1 <= d_ego + SYNC_TOLERANCE_S):
        raise ValueError(f"The selected eye-tracking part ({e0:.2f}-{e1:.2f} s) is outside the video.")
    if not (e0 + off >= -SYNC_TOLERANCE_S and e1 + off <= d_room + SYNC_TOLERANCE_S):
        raise ValueError(f"With an offset of {off:+.3f} s the selected part falls outside the room video.")
    e0 = max(e0, 0.0, -off)
    e1 = min(e1, d_ego, d_room - off)
    if e1 - e0 < 1.0:
        raise ValueError("The selected part is shorter than 1 second.")
    s.sync_mode = "segments"
    s.sync_offset_s = off
    s.ego_start_s, s.ego_end_s = e0, e1
    s.room_start_s, s.room_end_s = e0 + off, e1 + off
    if sync.get("audio_offset") is not None:
        s.audio_offset_s = float(sync["audio_offset"])
        s.audio_confidence = str(sync.get("audio_confidence") or "n/a")


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
            result["sync_used"] = {"offset": settings.sync_offset_s, "ego_start": settings.ego_start_s,
                                   "ego_end": settings.ego_end_s, "room_start": settings.room_start_s,
                                   "room_end": settings.room_end_s}
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
        return jsonify(error=str(exc)), 400
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
