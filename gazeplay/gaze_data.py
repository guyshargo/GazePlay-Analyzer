"""Load gaze data exported from Tobii Pro Lab (TSV / CSV / Excel).

Handles the variations seen in practice:
  * tab, semicolon or comma separated files, UTF-8 / UTF-16 / Latin-1
  * decimal commas (European locale exports)
  * timestamps in micro-, milli- or seconds, with or without units in the header
  * gaze in scene-video pixels ("Gaze point X [DACS px]") or normalised 0-1
  * several recordings in one export (the longest one is used)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TOBII_SCENE_SIZE = (1920, 1080)  # native scene-camera size of Tobii Pro Glasses 2/3

_TIME_PATTERNS = [
    r"^recording timestamp",
    r"^timestamp",
    r"^time ?stamp",
    r"^gaze timestamp",
    r"^eyetracker timestamp",
    r"^eye tracker timestamp",
    r"^computer timestamp",
    r"^time(\s*[\[(].*)?$",
    r"^time_?s(ec)?$",
]
def _axis_patterns(a: str) -> list[str]:
    return [
        rf"^gaze point {a}(\s*\[(dacs|scene|video)?\s*px\])?$",
        rf"^gaze point {a}(?!.*mcs)",
        rf"^gaze ?point ?2d[ _]?{a}\b", rf"^gaze ?2d[ _]?{a}\b", rf"^gaze[ _]{a}\b", rf"^gaze{a}$",
        rf"^gaze point {a}",          # last resort, even mapped (MCS) coordinates
        rf"^norm_pos_{a}$", rf"^{a}$",
    ]


_X_PATTERNS = _axis_patterns("x")
_Y_PATTERNS = _axis_patterns("y")
_FIX_X_PATTERNS = [r"^fixation point x(?!.*mcs)", r"^fixation point x"]
_FIX_Y_PATTERNS = [r"^fixation point y(?!.*mcs)", r"^fixation point y"]
_MOVEMENT_PATTERNS = [r"^eye movement type$", r"^eye movement type(?!.*index)", r"^gaze event type"]
_RECORDING_PATTERNS = [r"^recording name", r"^recording$"]


@dataclass
class GazeData:
    t: np.ndarray            # seconds, aligned to eye-tracking video time, sorted
    x: np.ndarray            # pixels in the eye-tracking video frame
    y: np.ndarray
    movement: np.ndarray | None
    info: dict = field(default_factory=dict)

    @property
    def start(self) -> float:
        return float(self.t[0]) if len(self.t) else 0.0

    @property
    def end(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def at(self, t: float, half_window: float):
        """Median gaze point within [t-hw, t+hw]; widens once to 2*hw. None if no data."""
        for hw in (half_window, 2 * half_window):
            i0 = np.searchsorted(self.t, t - hw, side="left")
            i1 = np.searchsorted(self.t, t + hw, side="right")
            if i1 > i0:
                xs, ys = self.x[i0:i1], self.y[i0:i1]
                mv = None
                if self.movement is not None:
                    j = i0 + int(np.argmin(np.abs(self.t[i0:i1] - t)))
                    mv = str(self.movement[j])
                return float(np.median(xs)), float(np.median(ys)), mv
        return None

    def valid_fraction(self, t0: float, t1: float, rate: float | None = None) -> float:
        rate = rate or self.info.get("sampling_rate_hz") or 50.0
        n = np.count_nonzero((self.t >= t0) & (self.t <= t1))
        expected = max(1.0, (t1 - t0) * rate)
        return float(min(1.0, n / expected))


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------
def _detect_text_format(path: Path):
    raw = path.read_bytes()[:256 * 1024]
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        enc = "utf-16"
    else:
        enc = None
        for cand in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                raw.decode(cand)
                enc = cand
                break
            except UnicodeDecodeError:
                continue
        enc = enc or "latin-1"
    text = raw.decode(enc, errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    counts = {"\t": header.count("\t"), ";": header.count(";"), ",": header.count(",")}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = "\t"
    return enc, sep


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=0, dtype=str)
    else:
        enc, sep = _detect_text_format(path)
        df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str, low_memory=False,
                         on_bad_lines="skip", engine="c")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _norm(col: str) -> str:
    return " ".join(str(col).strip().lower().replace("µ", "u").replace("μ", "u").split())


def _find(columns, patterns):
    normed = {c: _norm(c) for c in columns}
    for pat in patterns:
        rx = re.compile(pat)
        for c, n in normed.items():
            if rx.search(n):
                return c
    return None


def _to_float(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
    # decimal comma ("1234,56") -> "1234.56"; drop thousands separators/spaces
    s = s.str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    has_comma = s.str.contains(",", na=False)
    if has_comma.any():
        s = s.where(~has_comma, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _time_unit_factor(col: str, values: np.ndarray) -> tuple[float, str]:
    n = _norm(col)
    if re.search(r"\[\s*us\s*\]|\(\s*us\s*\)|micro", n):
        return 1e-6, "microseconds"
    if re.search(r"\[\s*ns\s*\]|\(\s*ns\s*\)", n):
        return 1e-9, "nanoseconds"
    if re.search(r"\[\s*ms\s*\]|\(\s*ms\s*\)", n):
        return 1e-3, "milliseconds"
    if re.search(r"\[\s*s\s*\]|\(\s*s\s*\)|_s$|sec", n):
        return 1.0, "seconds"
    v = values[np.isfinite(values)]
    if len(v) < 3:
        return 1e-3, "milliseconds (assumed)"
    d = np.diff(np.unique(v))
    med = float(np.median(d)) if len(d) else 1.0
    # Eye trackers sample at 25-1200 Hz: 0.8-40 ms between samples.
    if med > 2e5:
        return 1e-9, "nanoseconds (inferred)"
    if med > 200:
        return 1e-6, "microseconds (inferred)"
    if med > 0.2:
        return 1e-3, "milliseconds (inferred)"
    return 1.0, "seconds (inferred)"


def summarize_gaze_file(path: str | Path) -> dict:
    """Light-weight check used by the UI right after a file is dropped."""
    df = read_table(path)
    tcol = _find(df.columns, _TIME_PATTERNS)
    xcol = _find(df.columns, _X_PATTERNS)
    ycol = _find(df.columns, _Y_PATTERNS)
    out = {"rows": int(len(df)), "columns": len(df.columns), "time_column": tcol,
           "x_column": xcol, "y_column": ycol, "ok": bool(tcol and xcol and ycol)}
    if out["ok"]:
        t = _to_float(df[tcol])
        x = _to_float(df[xcol])
        f, unit = _time_unit_factor(tcol, t)
        t = t * f
        ok = np.isfinite(t)
        if ok.any():
            out["duration_s"] = float(np.nanmax(t) - np.nanmin(t))
        valid = np.isfinite(x) & ok
        out["valid_pct"] = float(100.0 * valid.sum() / max(1, ok.sum()))
        tv = np.unique(t[valid])
        if len(tv) > 3:
            out["sampling_rate_hz"] = float(round(1.0 / np.median(np.diff(tv))))
        out["time_unit"] = unit
    else:
        missing = [n for n, c in (("timestamp", tcol), ("gaze X", xcol), ("gaze Y", ycol)) if not c]
        out["error"] = "Could not find column(s): " + ", ".join(missing)
    return out


def load_gaze(path: str | Path, video_w: int, video_h: int, video_duration: float | None = None,
              fixations_only: bool = False, time_offset_s: float = 0.0) -> GazeData:
    df = read_table(path)
    cols = list(df.columns)
    info: dict = {"file": Path(path).name, "rows_in_file": int(len(df))}
    warnings: list[str] = []

    rec_col = _find(cols, _RECORDING_PATTERNS)
    if rec_col and df[rec_col].nunique(dropna=True) > 1:
        counts = df[rec_col].value_counts()
        chosen = counts.index[0]
        df = df[df[rec_col] == chosen]
        warnings.append(f"The gaze file contains {len(counts)} recordings; using '{chosen}' "
                        f"(most samples). Export one recording per file to choose another.")
        info["recording"] = str(chosen)

    tcol = _find(cols, _TIME_PATTERNS)
    xcol = _find(cols, _X_PATTERNS)
    ycol = _find(cols, _Y_PATTERNS)
    if not (tcol and xcol and ycol):
        missing = [n for n, c in (("timestamp", tcol), ("gaze X", xcol), ("gaze Y", ycol)) if not c]
        raise ValueError(
            "Could not recognise the gaze file. Missing column(s): " + ", ".join(missing)
            + ". Expected a Tobii Pro Lab export with 'Recording timestamp' and 'Gaze point X/Y'.")

    mcol = _find(cols, _MOVEMENT_PATTERNS)
    t = _to_float(df[tcol])
    x = _to_float(df[xcol])
    y = _to_float(df[ycol])
    movement = df[mcol].fillna("").astype(str).str.replace("nan", "", regex=False).to_numpy() if mcol else None

    if fixations_only:
        if movement is None:
            warnings.append("'Fixations only' was requested but the file has no 'Eye movement type' "
                            "column; all gaze samples were used.")
        else:
            fx = _find(cols, _FIX_X_PATTERNS)
            fy = _find(cols, _FIX_Y_PATTERNS)
            keep = np.char.lower(movement.astype(str)) == "fixation"
            if fx and fy:
                x = np.where(keep, _to_float(df[fx]), np.nan)
                y = np.where(keep, _to_float(df[fy]), np.nan)
                xcol, ycol = fx, fy
            else:
                x = np.where(keep, x, np.nan)
                y = np.where(keep, y, np.nan)

    factor, unit = _time_unit_factor(tcol, t)
    t = t * factor
    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        raise ValueError("The gaze file has fewer than 10 valid gaze samples.")
    t_all_min = float(np.nanmin(t))

    # Device-clock timestamps start far from zero -> make them relative.
    zeroed = False
    if video_duration and t_all_min > video_duration + 5:
        t = t - t_all_min
        zeroed = True
        warnings.append("Gaze timestamps did not start near 0 (device clock); they were shifted so "
                        "the first sample is at 0 s. Use 'Gaze time offset' if this is wrong.")

    t, x, y = t[valid], x[valid], y[valid]
    if movement is not None:
        movement = movement[valid]
    order = np.argsort(t, kind="stable")
    t, x, y = t[order] + time_offset_s, x[order], y[order]
    if movement is not None:
        movement = movement[order]

    # Coordinate system
    xs_abs = np.nanpercentile(np.abs(x), 99)
    ys_abs = np.nanpercentile(np.abs(y), 99)
    if "norm" in _norm(xcol) or (xs_abs <= 1.5 and ys_abs <= 1.5):
        x, y = x * video_w, y * video_h
        coord = "normalised (0-1) -> scaled to video size"
    else:
        if xs_abs <= video_w * 1.15 and ys_abs <= video_h * 1.15:
            ref_w, ref_h = video_w, video_h
        else:
            ref_w, ref_h = TOBII_SCENE_SIZE
            if (ref_w, ref_h) != (video_w, video_h):
                warnings.append(f"Gaze coordinates exceed the video size; assumed they refer to the "
                                f"{ref_w}x{ref_h} Tobii scene camera and rescaled them.")
        x = x * (video_w / ref_w)
        y = y * (video_h / ref_h)
        coord = f"pixels ({ref_w}x{ref_h})" + ("" if (ref_w, ref_h) == (video_w, video_h)
                                              else f" -> rescaled to {video_w}x{video_h}")

    ut = np.unique(t)
    rate = float(1.0 / np.median(np.diff(ut))) if len(ut) > 3 else 50.0
    info.update({
        "time_column": tcol, "x_column": xcol, "y_column": ycol,
        "movement_column": mcol, "time_unit": unit, "timestamps_zeroed": zeroed,
        "coordinates": coord, "valid_samples": int(len(t)),
        "sampling_rate_hz": round(rate, 1),
        "first_s": round(float(t[0]), 3), "last_s": round(float(t[-1]), 3),
        "fixations_only": bool(fixations_only and movement is not None),
        "warnings": warnings,
    })
    return GazeData(t=t, x=x, y=y, movement=movement, info=info)
