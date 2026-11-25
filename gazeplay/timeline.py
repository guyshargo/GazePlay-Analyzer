"""Combine both streams on one time axis: smoothing, matching, play bouts, statistics."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .config import (CHILD_NOT_VISIBLE, HANDS_NOT_VISIBLE, MULTI_SEP, NO_GAZE, NON_OBJECT_GAZE,
                     NON_OBJECT_INTERACTION, OUTSIDE_VIEW)

COL_TS = "Timestamp"
COL_INTER = "Interacted Object"
COL_GAZE = "Gazed-at Object"
COL_MATCH = "Looking at Played Object"
COL_T = "Time (s)"
COL_TROOM = "Room Video Time (s)"


def fmt_ts(t: float) -> str:
    t = max(0.0, float(t))
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def mode_filter(labels: list[str], win: int) -> list[str]:
    """Majority vote in a centred window; keeps the current label on ties."""
    n = len(labels)
    if win <= 1 or n == 0:
        return list(labels)
    half = win // 2
    out = []
    for i in range(n):
        cnt = Counter(labels[max(0, i - half): i + half + 1])
        top = max(cnt.values())
        out.append(labels[i] if cnt[labels[i]] == top else cnt.most_common(1)[0][0])
    return out


def interaction_objects(label: str) -> list[str]:
    if not label or label in NON_OBJECT_INTERACTION:
        return []
    return label.split(MULTI_SEP)


def match_status(inter: str, gaze: str) -> str:
    objs = interaction_objects(inter)
    if not objs or gaze in (NO_GAZE, OUTSIDE_VIEW):  # unknown: we cannot see what was looked at
        return "n/a"
    return "Yes" if gaze in objs else "No"


def build_timeline(times, room_samples, gaze_samples, offset_s: float, settings) -> pd.DataFrame:
    dt = 1.0 / settings.sample_rate
    raw_inter = [s.label for s in room_samples]
    raw_gaze = [s.label for s in gaze_samples]
    w_inter = max(1, int(round(settings.smooth_interaction_s / dt)) | 1)  # odd window
    w_gaze = max(1, int(round(settings.smooth_gaze_s / dt)) | 1)
    inter = mode_filter(raw_inter, w_inter)
    gaze = mode_filter(raw_gaze, w_gaze)
    rows = []
    for i, t in enumerate(times):
        rs, gs = room_samples[i], gaze_samples[i]
        rows.append({
            COL_TS: fmt_ts(t),
            COL_INTER: inter[i],
            COL_GAZE: gaze[i],
            COL_MATCH: match_status(inter[i], gaze[i]),
            COL_T: round(float(t), 3),
            COL_TROOM: round(float(t + offset_s), 3),
            "Gaze X (px)": None if gs.gaze is None else round(gs.gaze[0], 1),
            "Gaze Y (px)": None if gs.gaze is None else round(gs.gaze[1], 1),
            "Eye Movement": gs.movement or "",
            "Interaction Confidence": round(rs.confidence, 2),
            "Gaze Confidence": round(gs.confidence, 2),
            "Interacted (unsmoothed)": raw_inter[i],
            "Gazed-at (unsmoothed)": raw_gaze[i],
        })
    return pd.DataFrame(rows)


def segments(values: list[str], times: list[float], dt: float) -> list[dict]:
    """Run-length encode a label sequence -> [{start, end, label}]."""
    out = []
    for i, v in enumerate(values):
        if out and out[-1]["label"] == v and abs(out[-1]["end"] - times[i]) < 1.5 * dt:
            out[-1]["end"] = round(times[i] + dt, 3)
        else:
            out.append({"start": round(times[i], 3), "end": round(times[i] + dt, 3), "label": v})
    return out


def play_bouts(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    labels = df[COL_INTER].tolist()
    gazes = df[COL_GAZE].tolist()
    times = df[COL_T].tolist()
    bouts, start = [], None
    for i in range(len(labels) + 1):
        cur = labels[i] if i < len(labels) else None
        is_obj = bool(cur and interaction_objects(cur))
        if start is not None and (not is_obj or cur != labels[start]):
            bouts.append((start, i - 1))
            start = None
        if is_obj and start is None:
            start = i
    rows = []
    for n, (a, b) in enumerate(bouts, 1):
        objs = interaction_objects(labels[a])
        g = gazes[a: b + 1]
        valid = [x for x in g if x not in (NO_GAZE, OUTSIDE_VIEW)]
        looking = [x for x in valid if x in objs]
        first = next((times[a + j] - times[a] for j, x in enumerate(g) if x in objs), None)
        others = Counter(x for x in valid if x not in objs)
        other_txt = ", ".join(f"{k} ({100 * v / len(valid):.0f}%)" for k, v in others.most_common(3)) if valid else ""
        rows.append({
            "Bout": n,
            "Start": fmt_ts(times[a]),
            "End": fmt_ts(times[b] + dt),
            "Duration (s)": round((b - a + 1) * dt, 2),
            "Object(s) Played With": labels[a],
            "% Time Looking at It": round(100 * len(looking) / len(valid), 1) if valid else None,
            "Looked at It?": "Yes" if looking else ("No" if valid else "n/a"),
            "Time to First Look (s)": None if first is None else round(first, 2),
            "Gaze Data Coverage (%)": round(100 * len(valid) / len(g), 1),  # gaze inside the camera view
            "Other Things Looked At": other_txt,
            "Start (s)": round(times[a], 3),
        })
    cols = ["Bout", "Start", "End", "Duration (s)", "Object(s) Played With", "% Time Looking at It",
            "Looked at It?", "Time to First Look (s)", "Gaze Data Coverage (%)",
            "Other Things Looked At", "Start (s)"]
    return pd.DataFrame(rows, columns=cols)


def per_object_stats(df: pd.DataFrame, bouts: pd.DataFrame, dt: float, vocabulary: list[str],
                     room_samples, gaze_samples) -> pd.DataFrame:
    inter_sets = [set(interaction_objects(x)) for x in df[COL_INTER]]
    gazes = df[COL_GAZE].tolist()
    seen = [lab for lab in vocabulary]
    for s in inter_sets:
        seen += [x for x in s if x not in seen]
    seen += [g for g in dict.fromkeys(gazes) if g not in seen and g not in NON_OBJECT_GAZE]
    n = max(1, len(df))
    room_det = Counter()
    for s in room_samples:
        for lab in {d.label for d in s.objects}:
            room_det[lab] += 1
    gaze_det = Counter()
    n_gaze_frames = max(1, sum(1 for s in gaze_samples if s.gaze is not None))
    for s in gaze_samples:
        for lab in {d.label for d in s.objects}:
            gaze_det[lab] += 1
    bout_counts = Counter()
    for lab in bouts.get("Object(s) Played With", []):
        for o in interaction_objects(lab):
            bout_counts[o] += 1
    rows = []
    for lab in seen:
        play = [i for i, s in enumerate(inter_sets) if lab in s]
        play_valid = [i for i in play if gazes[i] not in (NO_GAZE, OUTSIDE_VIEW)]
        look_play = [i for i in play_valid if gazes[i] == lab]
        gaze_n = sum(1 for g in gazes if g == lab)
        rows.append({
            "Object": lab,
            "Play Time (s)": round(len(play) * dt, 1),
            "Play Bouts": bout_counts.get(lab, 0),
            "Looking-While-Playing Time (s)": round(len(look_play) * dt, 1),
            "% of Play Time Looking at It": round(100 * len(look_play) / len(play_valid), 1) if play_valid else None,
            "Total Gaze Time (s)": round(gaze_n * dt, 1),
            "% of Session Gazed": round(100 * gaze_n / n, 1),
            "Detected in Room Video (% samples)": round(100 * room_det.get(lab, 0) / n, 1),
            "Detected Around Gaze (% samples)": round(100 * gaze_det.get(lab, 0) / n_gaze_frames, 1),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Play Time (s)", "Total Gaze Time (s)"], ascending=False).reset_index(drop=True)
    return out


def summary_metrics(df: pd.DataFrame, bouts: pd.DataFrame, dt: float) -> dict:
    n = len(df)
    inter = df[COL_INTER].tolist()
    gaze = df[COL_GAZE].tolist()
    child_vis = sum(1 for x in inter if x != CHILD_NOT_VISIBLE)
    hands_vis = sum(1 for x in inter if x not in (CHILD_NOT_VISIBLE, HANDS_NOT_VISIBLE))
    playing = [i for i, x in enumerate(inter) if interaction_objects(x)]
    playing_valid = [i for i in playing if gaze[i] not in (NO_GAZE, OUTSIDE_VIEW)]
    yes = sum(1 for i in playing_valid if gaze[i] in interaction_objects(inter[i]))
    gaze_valid = sum(1 for g in gaze if g != NO_GAZE)
    on_obj = sum(1 for g in gaze if g not in NON_OBJECT_GAZE)
    outside = sum(1 for g in gaze if g == OUTSIDE_VIEW)
    play_counter = Counter(o for i in playing for o in interaction_objects(inter[i]))
    gaze_counter = Counter(g for g in gaze if g not in NON_OBJECT_GAZE)
    pct = lambda a, b: round(100.0 * a / b, 1) if b else None  # noqa: E731
    return {
        "analysed_from": fmt_ts(df[COL_T].iloc[0]) if n else "",
        "analysed_to": fmt_ts(df[COL_T].iloc[-1] + dt) if n else "",
        "duration_s": round(n * dt, 1),
        "samples": n,
        "child_visible_pct": pct(child_vis, n),
        "hands_visible_pct": pct(hands_vis, n),
        "playing_time_s": round(len(playing) * dt, 1),
        "playing_pct": pct(len(playing), n),
        "gaze_valid_pct": pct(gaze_valid, n),
        "gaze_on_object_pct": pct(on_obj, gaze_valid),
        "gaze_outside_view_pct": pct(outside, gaze_valid),
        "looking_at_played_pct": pct(yes, len(playing_valid)),
        "looking_at_played_time_s": round(yes * dt, 1),
        "play_bouts": int(len(bouts)),
        "mean_bout_s": round(float(bouts["Duration (s)"].mean()), 2) if len(bouts) else None,
        "bouts_with_look_pct": pct(int((bouts["Looked at It?"] == "Yes").sum()), int((bouts["Looked at It?"] != "n/a").sum())) if len(bouts) else None,
        "most_played": play_counter.most_common(1)[0][0] if play_counter else None,
        "most_looked": gaze_counter.most_common(1)[0][0] if gaze_counter else None,
    }


# Validated categorical palette (colour-blind safe in this fixed order). Objects beyond the
# eighth share a neutral "other" colour rather than receiving generated hues.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOR = "#8a8984"


def color_palette(labels: list[str]) -> dict[str, str]:
    """Stable colour per object, in list order (hex), shared by charts and the review video."""
    return {lab: (CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER_COLOR) for i, lab in enumerate(labels)}


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
