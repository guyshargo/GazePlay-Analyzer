"""Excel / CSV / JSON output."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import (APP_NAME, APP_VERSION, CHILD_NOT_VISIBLE, HANDS_NOT_VISIBLE, NO_GAZE,
                     NO_INTERACTION, NO_OBJECT)
from .timeline import COL_MATCH

LABEL_HELP = [
    ("Interacted Object", "<object name>", "The child's hand is on / holding this object in the room video."),
    ("Interacted Object", "a + b", "Two objects are in the child's hands at once."),
    ("Interacted Object", NO_INTERACTION, "Child and hands are visible, but no object is in hand."),
    ("Interacted Object", HANDS_NOT_VISIBLE, "Child visible, but both hands are hidden from the room camera."),
    ("Interacted Object", CHILD_NOT_VISIBLE, "No child found in the room video at this moment."),
    ("Gazed-at Object", "<object name>", "The gaze point falls on (or within the accuracy margin of) this object."),
    ("Gazed-at Object", NO_OBJECT, "Valid gaze, but not on any of the listed objects."),
    ("Gazed-at Object", NO_GAZE, "No valid gaze sample (blink, look-away, tracking loss)."),
    (COL_MATCH, "Yes / No", "Whether the gazed-at object is the object being played with."),
    (COL_MATCH, "n/a", "Not applicable: the child is not playing with an object, or there is no gaze data."),
]


def _fmt_pct(v):
    return "-" if v is None else f"{v:.1f}%"


def summary_rows(metrics: dict) -> list[tuple[str, str]]:
    m = metrics
    rows = [
        ("Session analysed", f"{m['analysed_from']} - {m['analysed_to']}  ({m['duration_s']:.1f} s, {m['samples']} samples)"),
        ("Time playing with an object", f"{m['playing_time_s']:.1f} s  ({_fmt_pct(m['playing_pct'])} of session)"),
        ("While playing: looking at the played object", f"{_fmt_pct(m['looking_at_played_pct'])} of play time with gaze data  ({m['looking_at_played_time_s']:.1f} s)"),
        ("Play bouts", f"{m['play_bouts']}" + (f"  (mean {m['mean_bout_s']:.1f} s)" if m.get('mean_bout_s') else "")),
        ("Bouts with at least one look at the object", _fmt_pct(m.get("bouts_with_look_pct"))),
        ("Most played-with object", m.get("most_played") or "-"),
        ("Most looked-at object", m.get("most_looked") or "-"),
        ("Valid gaze data", _fmt_pct(m["gaze_valid_pct"])),
        ("Gaze on a listed object (of valid gaze)", _fmt_pct(m["gaze_on_object_pct"])),
        ("Child visible in room video", _fmt_pct(m["child_visible_pct"])),
        ("Child's hands visible", _fmt_pct(m["hands_visible_pct"])),
    ]
    return rows


def write_excel(path: Path, timeline: pd.DataFrame, bouts: pd.DataFrame, per_object: pd.DataFrame,
                objects_found: pd.DataFrame, metrics: dict, context: dict) -> Path:
    path = Path(path)
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        wb = xw.book
        f_title = wb.add_format({"bold": True, "font_size": 16, "font_color": "#1d3557"})
        f_h2 = wb.add_format({"bold": True, "font_size": 12, "font_color": "#1d3557", "bottom": 1})
        f_key = wb.add_format({"bold": True, "valign": "top"})
        f_wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        f_muted = wb.add_format({"font_color": "#666666", "italic": True})
        f_head = wb.add_format({"bold": True, "bg_color": "#1d3557", "font_color": "#ffffff",
                                "text_wrap": True, "valign": "vcenter", "border": 1})
        f_yes = wb.add_format({"bg_color": "#d4f1df", "font_color": "#0b6b35"})
        f_no = wb.add_format({"bg_color": "#fbe0dc", "font_color": "#9c2a1b"})
        f_na = wb.add_format({"font_color": "#999999"})
        f_big = wb.add_format({"bold": True, "font_size": 12, "valign": "top", "text_wrap": True})

        # ---------------- Summary ----------------
        ws = wb.add_worksheet("Summary")
        xw.sheets["Summary"] = ws
        ws.set_column(0, 0, 46)
        ws.set_column(1, 1, 90)
        ws.set_landscape()
        ws.fit_to_pages(1, 0)
        ws.hide_gridlines(2)
        r = 0
        ws.write(r, 0, f"{APP_NAME} - Results", f_title); r += 1
        ws.write(r, 0, f"Generated {datetime.now():%Y-%m-%d %H:%M}  |  version {APP_VERSION}", f_muted); r += 2
        ws.write(r, 0, "Key results", f_h2); ws.write(r, 1, "", f_h2); r += 1
        for i, (k, v) in enumerate(summary_rows(metrics)):
            ws.write(r, 0, k, f_key)
            ws.write(r, 1, v, f_big if i in (1, 2) else f_wrap)
            r += 1
        for title, items in context.items():
            r += 1
            ws.write(r, 0, title, f_h2); ws.write(r, 1, "", f_h2); r += 1
            if not items:
                ws.write(r, 0, "None", f_muted); r += 1
            for k, v in items:
                ws.write(r, 0, k, f_key)
                ws.write(r, 1, v, f_wrap)
                r += 1
        r += 1
        ws.write(r, 0, "What the labels mean", f_h2); ws.write(r, 1, "", f_h2); r += 1
        for col, lab, meaning in LABEL_HELP:
            ws.write(r, 0, f"{col}: {lab}", f_key)
            ws.write(r, 1, meaning, f_wrap)
            r += 1

        # ---------------- Timeline ----------------
        def table(df: pd.DataFrame, name: str, widths: dict | None = None):
            df.to_excel(xw, sheet_name=name, index=False, startrow=0)
            sh = xw.sheets[name]
            for j, col in enumerate(df.columns):
                sh.write(0, j, col, f_head)
                w = (widths or {}).get(col)
                if w is None:
                    sample = df[col].head(500).map(lambda v: len(str(v)) if v is not None else 0)
                    w = min(45, max(10, int(max(len(col) * 0.9, sample.max() if len(sample) else 0) + 2)))
                sh.set_column(j, j, w)
            sh.set_row(0, 32)
            sh.freeze_panes(1, 1)
            sh.set_landscape()
            sh.fit_to_pages(1, 0)
            sh.repeat_rows(0)
            if len(df):
                sh.autofilter(0, 0, len(df), len(df.columns) - 1)
            return sh

        sh = table(timeline, "Timeline", {"Timestamp": 14, "Interacted Object": 22, "Gazed-at Object": 22,
                                          COL_MATCH: 14})
        if len(timeline):
            j = list(timeline.columns).index(COL_MATCH)
            rng = (1, j, len(timeline), j)
            sh.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"Yes"', "format": f_yes})
            sh.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"No"', "format": f_no})
            sh.conditional_format(*rng, {"type": "cell", "criteria": "==", "value": '"n/a"', "format": f_na})

        # ---------------- Play bouts ----------------
        sh = table(bouts, "Play Bouts", {"Other Things Looked At": 45, "Object(s) Played With": 24})
        if len(bouts):
            j = list(bouts.columns).index("% Time Looking at It")
            sh.conditional_format(1, j, len(bouts), j, {"type": "data_bar", "bar_color": "#7fb8e6",
                                                        "min_type": "num", "min_value": 0,
                                                        "max_type": "num", "max_value": 100})
            j = list(bouts.columns).index("Looked at It?")
            sh.conditional_format(1, j, len(bouts), j, {"type": "cell", "criteria": "==", "value": '"Yes"', "format": f_yes})
            sh.conditional_format(1, j, len(bouts), j, {"type": "cell", "criteria": "==", "value": '"No"', "format": f_no})

        # ---------------- Per object (+ chart) ----------------
        sh = table(per_object, "Per Object", {"Object": 22})
        n = len(per_object)
        if n:
            j = list(per_object.columns).index("% of Play Time Looking at It")
            sh.conditional_format(1, j, n, j, {"type": "data_bar", "bar_color": "#7fb8e6",
                                               "min_type": "num", "min_value": 0,
                                               "max_type": "num", "max_value": 100})
            chart = wb.add_chart({"type": "bar"})
            for col, color in (("Play Time (s)", "#9db9d9"), ("Looking-While-Playing Time (s)", "#1d3557")):
                c = list(per_object.columns).index(col)
                chart.add_series({"name": ["Per Object", 0, c],
                                  "categories": ["Per Object", 1, 0, n, 0],
                                  "values": ["Per Object", 1, c, n, c],
                                  "fill": {"color": color}, "gap": 60})
            chart.set_title({"name": "Play time and looking-while-playing, per object"})
            chart.set_x_axis({"name": "seconds"})
            chart.set_y_axis({"reverse": True})
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": 720, "height": max(260, 40 * n + 120)})
            sh.insert_chart(n + 3, 0, chart)

        # ---------------- Objects found ----------------
        if objects_found is not None and len(objects_found):
            table(objects_found, "Objects Found")
    return path


def write_csv(path: Path, timeline: pd.DataFrame) -> Path:
    timeline.to_csv(path, index=False, encoding="utf-8-sig")
    return Path(path)


def write_json(path: Path, payload: dict) -> Path:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return Path(path)
