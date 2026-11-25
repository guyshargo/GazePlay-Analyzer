"""Command-line / batch version of GazePlay Analyzer.

One session:
    python analyze.py --ego scene.mp4 --room room.mp4 --gaze gaze.tsv --objects "ball, teddy bear"

Many sessions (one row per session; columns ego, room, gaze and optionally objects, offset, out):
    python analyze.py --manifest sessions.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gazeplay.config import PRESETS, RESULTS_DIR, Settings, parse_object_list  # noqa: E402
from gazeplay.pipeline import STAGES, run_analysis  # noqa: E402

STAGE_NAMES = {k: label for k, label, _ in STAGES}


def console_progress():
    last = {"line": ""}

    def cb(stage=None, stage_frac=0.0, overall=0.0, message="", eta_s=None, logs=None):
        bar = "#" * int(30 * overall) + "-" * (30 - int(30 * overall))
        eta = f" | ~{eta_s / 60:.0f} min left" if eta_s and eta_s > 60 else ""
        line = f"\r[{bar}] {100 * overall:5.1f}%  {STAGE_NAMES.get(stage, '')[:48]:<48}{eta}"
        if line != last["line"]:
            sys.stdout.write(line)
            sys.stdout.flush()
            last["line"] = line
    return cb


def build_settings(args, objects=None, offset=None) -> Settings:
    s = Settings.from_preset(args.preset)
    s.objects = parse_object_list(objects if objects is not None else args.objects)
    if offset not in (None, ""):
        s.sync_mode, s.manual_offset_s = "manual", float(offset)
    elif args.offset is not None:
        s.sync_mode, s.manual_offset_s = "manual", float(args.offset)
    elif args.no_sync:
        s.sync_mode = "none"
    if args.ego_start is not None and args.room_start is not None:
        s.sync_mode = "segments"
        s.ego_start_s, s.room_start_s = args.ego_start, args.room_start
        s.ego_end_s, s.room_end_s = args.ego_end, args.room_end
    elif any(v is not None for v in (args.ego_start, args.room_start, args.ego_end, args.room_end)):
        raise SystemExit("--ego-start and --room-start must be given together")
    s.make_review_video = not args.no_video
    s.include_people = args.include_people
    s.fixations_only = args.fixations_only
    if args.max_seconds:
        s.max_duration_s = args.max_seconds
    return s


def run_one(ego, room, gaze, out, settings, backend=None):
    t = time.time()
    ego, room, gaze, out = (str(Path(x).resolve()) for x in (ego, room, gaze, out))
    res = run_analysis(ego, room, gaze, out, settings, progress_cb=console_progress(), backend=backend)
    m = res["metrics"]
    print(f"\n  Done in {(time.time() - t) / 60:.1f} min.  Sync offset {res['sync']['offset_s']:+.2f} s "
          f"({res['sync']['confidence']}).")
    print(f"  Playing with objects: {m['playing_time_s']:.1f} s | looking at the played object: "
          f"{m['looking_at_played_pct'] if m['looking_at_played_pct'] is not None else '-'}%")
    for w in res["warnings"]:
        print("  ! " + w)
    print(f"  Excel: {res['files']['excel']}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Did the child look at the object they were playing with?")
    ap.add_argument("--ego", help="first-person video from the eye-tracking glasses")
    ap.add_argument("--room", help="external room-camera video")
    ap.add_argument("--gaze", help="Tobii Pro Lab gaze export (.tsv/.csv/.xlsx)")
    ap.add_argument("--manifest", help="CSV/Excel with columns ego, room, gaze [, objects, offset, out]")
    ap.add_argument("--objects", default="", help='comma-separated objects, e.g. "red ball, teddy bear"')
    ap.add_argument("--preset", default="balanced", choices=list(PRESETS))
    ap.add_argument("--offset", type=float, help="manual sync offset in s (room time - glasses time)")
    ap.add_argument("--no-sync", action="store_true", help="videos are already synchronised")
    ap.add_argument("--ego-start", type=float, help="segment start in the eye-tracking video (s); "
                    "with --room-start this sets the sync: both starts are the same moment")
    ap.add_argument("--ego-end", type=float, help="segment end in the eye-tracking video (s)")
    ap.add_argument("--room-start", type=float, help="segment start in the room video (s)")
    ap.add_argument("--room-end", type=float, help="segment end in the room video (s)")
    ap.add_argument("--no-video", action="store_true", help="skip the review video")
    ap.add_argument("--include-people", action="store_true", help="count people (e.g. a parent's face) as gaze targets")
    ap.add_argument("--fixations-only", action="store_true")
    ap.add_argument("--max-seconds", type=float, help="analyse only the first N seconds")
    ap.add_argument("--out", help="output folder")
    args = ap.parse_args()

    if args.manifest:
        import pandas as pd
        path = Path(args.manifest)
        df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        backend = None
        for i, row in df.iterrows():
            print(f"\n=== Session {i + 1}/{len(df)}: {row['ego']}")
            out = row.get("out") if isinstance(row.get("out"), str) else \
                RESULTS_DIR / f"{Path(str(row['ego'])).stem}_{datetime.now():%Y-%m-%d_%H%M%S}"
            s = build_settings(args, row.get("objects") if isinstance(row.get("objects"), str) else None,
                               row.get("offset") if "offset" in row and pd.notna(row.get("offset")) else None)
            if backend is None:
                from gazeplay.detectors import UltralyticsBackend
                backend = UltralyticsBackend(log=lambda m: None)
            try:
                run_one(str(row["ego"]), str(row["room"]), str(row["gaze"]), out, s, backend)
            except Exception as exc:
                print(f"\n  FAILED: {exc}")
        return

    if not (args.ego and args.room and args.gaze):
        ap.error("give --ego, --room and --gaze (or --manifest)")
    out = args.out or RESULTS_DIR / f"{Path(args.ego).stem}_{datetime.now():%Y-%m-%d_%H%M%S}"
    run_one(args.ego, args.room, args.gaze, out, build_settings(args))


if __name__ == "__main__":
    main()
