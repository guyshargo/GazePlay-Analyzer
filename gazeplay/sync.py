"""Automatic synchronisation of the two videos from their sound tracks.

Both cameras record the same room sounds (speech, toys, a clap). We turn each
sound track into an "onset envelope" (how sharply loudness rises every 5 ms),
then cross-correlate the two envelopes. The lag with the strongest match is the
time offset between the recordings. Envelopes are robust to the two
microphones having different gain and frequency response.

Convention used everywhere in the app:
    room_video_time = eye_tracking_video_time + offset_s
(a positive offset means the room camera started recording first).
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

AUDIO_SR = 8000
ENV_HOP_S = 0.005  # 200 envelope samples per second


@dataclass
class SyncResult:
    offset_s: float
    method: str          # "audio" | "manual" | "none"
    confidence: str      # "high" | "medium" | "low" | "n/a"
    score: float = 0.0   # peak z-score
    message: str = ""

    def to_dict(self):
        return {"offset_s": round(self.offset_s, 3), "method": self.method,
                "confidence": self.confidence, "score": round(self.score, 1),
                "message": self.message}


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg  # bundled binary, works on Windows without setup
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def extract_audio(path: str, sr: int = AUDIO_SR, max_seconds: float | None = None) -> np.ndarray | None:
    exe = ffmpeg_exe()
    if not exe:
        return None
    cmd = [exe, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    except Exception:
        return None
    if proc.returncode != 0 or len(proc.stdout) < sr * 2 * 2:  # < 2 s of audio
        return None
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if float(np.max(np.abs(audio))) < 1e-4:  # silent track
        return None
    return audio


def onset_envelope(audio: np.ndarray, sr: int = AUDIO_SR, hop_s: float = ENV_HOP_S) -> np.ndarray:
    hop = max(1, int(round(sr * hop_s)))
    x = np.empty_like(audio)
    x[0] = audio[0]
    x[1:] = audio[1:] - 0.97 * audio[:-1]          # pre-emphasis: favour transients
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    energy = np.log10(np.mean(frames * frames, axis=1) + 1e-9)
    onset = np.diff(energy, prepend=energy[0])
    onset = np.maximum(onset, 0.0)
    # local whitening over ~2 s so loud and quiet passages count equally
    win = int(2.0 / hop_s)
    kernel = np.ones(win) / win
    mean = np.convolve(onset, kernel, mode="same")
    sq = np.convolve(onset * onset, kernel, mode="same")
    std = np.sqrt(np.maximum(sq - mean * mean, 1e-12))
    env = (onset - mean) / (std + 1e-3)
    return np.clip(env, -3, 10).astype(np.float64)


def cross_correlate_offset(env_ego: np.ndarray, env_room: np.ndarray, rate: float,
                           max_lag_s: float | None = None, min_overlap_s: float = 15.0):
    """Return (offset_s, z_score, peak_ratio) maximising sum_n ego[n] * room[n + lag]."""
    a = env_ego - env_ego.mean()
    b = env_room - env_room.mean()
    na, nb = len(a), len(b)
    nfft = 1 << int(np.ceil(np.log2(na + nb)))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
    lags = np.arange(-(na - 1), nb)
    vals = c[lags % nfft]
    overlap = np.minimum(na, nb - lags) - np.maximum(0, -lags)
    min_overlap = max(int(min_overlap_s * rate), int(0.2 * min(na, nb)))
    ok = overlap >= max(min_overlap, 1)
    if max_lag_s is not None:
        ok &= np.abs(lags) <= max_lag_s * rate
    if not ok.any():
        return 0.0, 0.0, 1.0
    score = np.full(len(lags), -np.inf)
    score[ok] = vals[ok] / np.sqrt(overlap[ok])
    best = int(np.argmax(score))
    valid_scores = score[ok]
    med = float(np.median(valid_scores))
    mad = float(np.median(np.abs(valid_scores - med))) * 1.4826 + 1e-9
    z = (score[best] - med) / mad
    # second-best peak at least 0.5 s away
    guard = int(0.5 * rate)
    masked = score.copy()
    masked[max(0, best - guard): best + guard + 1] = -np.inf
    second = float(np.max(masked)) if np.isfinite(masked).any() else med
    ratio = float((score[best] - med) / max(second - med, 1e-9))
    # parabolic interpolation for sub-frame precision
    lag = float(lags[best])
    if 0 < best < len(score) - 1 and np.isfinite(score[best - 1]) and np.isfinite(score[best + 1]):
        y0, y1, y2 = score[best - 1], score[best], score[best + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            lag += 0.5 * (y0 - y2) / denom
    return lag / rate, float(z), ratio


def fmt_clock(t: float) -> str:
    t = max(0.0, float(t))
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return (f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}")


def offset_from_envelopes(e1: np.ndarray, e2: np.ndarray):
    """(offset_s, confidence, z) for room = ego + offset."""
    offset, z, ratio = cross_correlate_offset(e1, e2, 1.0 / ENV_HOP_S)
    if z >= 12 and ratio >= 1.25:
        conf = "high"
    elif z >= 7 and ratio >= 1.1:
        conf = "medium"
    else:
        conf = "low"
    return offset, conf, z


def estimate_offset_from_audio(ego_video: str, room_video: str, log=lambda m: None) -> SyncResult:
    ego = extract_audio(ego_video)
    room = extract_audio(room_video)
    if ego is None or room is None:
        which = "eye-tracking video" if ego is None else "room video"
        return SyncResult(0.0, "none", "n/a", 0.0,
                          f"No usable audio track in the {which}; assumed both videos start together "
                          f"(offset 0 s). Enter the offset manually under Advanced settings if needed.")
    offset, conf, z = offset_from_envelopes(onset_envelope(ego), onset_envelope(room))
    who = "room camera started" if offset >= 0 else "eye-tracking glasses started"
    msg = f"Audio sync: {who} {abs(offset):.2f} s earlier (confidence: {conf})."
    if conf == "low":
        msg += (" The sound tracks did not match clearly - please check the review video and, if "
                "needed, enter the offset manually under Advanced settings.")
    log(msg)
    return SyncResult(float(offset), "audio", conf, z, msg)


def resolve_sync(settings, ego_video: str, room_video: str, log=lambda m: None) -> SyncResult:
    mode = (settings.sync_mode or "audio").lower()
    if mode == "segments":
        e0, r0 = float(settings.ego_start_s or 0.0), float(settings.room_start_s or 0.0)
        off = float(settings.sync_offset_s) if settings.sync_offset_s is not None else r0 - e0
        msg = f"Confirmed by the user: offset {off:+.3f} s (eye-tracking {fmt_clock(e0)} = room {fmt_clock(r0)})."
        if settings.audio_offset_s is not None:
            adj = off - float(settings.audio_offset_s)
            msg += (f" Automatic audio sync found {float(settings.audio_offset_s):+.3f} s"
                    f" ({settings.audio_confidence or 'n/a'} confidence)"
                    + (f"; adjusted by hand by {adj:+.3f} s." if abs(adj) >= 0.0005 else "; used unchanged."))
        conf = settings.audio_confidence if settings.audio_offset_s is not None else "n/a"
        return SyncResult(off, "segments", conf or "n/a", 0.0, msg)
    if mode == "manual":
        return SyncResult(float(settings.manual_offset_s), "manual", "n/a", 0.0,
                          f"Manual offset: {settings.manual_offset_s:+.2f} s.")
    if mode == "none":
        return SyncResult(0.0, "none", "n/a", 0.0, "Videos treated as already synchronised (offset 0 s).")
    return estimate_offset_from_audio(ego_video, room_video, log)
