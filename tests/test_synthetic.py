"""End-to-end test on synthetic data with the colour mock detector.

    python tests/test_synthetic.py            (generates data on first run)
    pytest tests/test_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gazeplay.config import Settings  # noqa: E402
from gazeplay.pipeline import run_analysis  # noqa: E402
from make_synthetic import OFFSET, ground_truth, main as make_data  # noqa: E402
from mock_backend import MockBackend  # noqa: E402

DATA = ROOT / "tests" / "synthetic"


def run(objects=("ball", "block", "toy car"), out=None, **kw):
    if not (DATA / "gaze_export.tsv").exists():
        make_data(DATA)
    s = Settings.from_preset("balanced", objects=list(objects), **kw)
    return run_analysis(str(DATA / "tobii_scene.mp4"), str(DATA / "room_camera.mp4"),
                        str(DATA / "gaze_export.tsv"), out or (ROOT / "tests" / "out"), s,
                        backend=MockBackend())


def accuracy(result):
    df = pd.read_csv(result["files"]["csv"])
    gt = {round(t, 1): (i, g) for t, i, g in ground_truth()}
    n = ok_i = ok_g = 0
    for row in df.itertuples(index=False):
        t = round(row[4], 1)
        if t not in gt:
            continue
        # ignore 0.6 s around scripted transitions (smoothing window)
        if any(abs(t - b) < 0.6 for b in (0, 10, 12, 20, 30, 30.6, 40, 55, 61, 62)):
            continue
        n += 1
        ok_i += row[1] == gt[t][0]
        ok_g += row[2] == gt[t][1]
    return ok_i / n, ok_g / n, n


def test_end_to_end():
    res = run()
    assert abs(res["sync"]["offset_s"] - OFFSET) < 0.05
    acc_i, acc_g, n = accuracy(res)
    assert n > 250
    assert acc_i > 0.97, acc_i
    assert acc_g > 0.97, acc_g
    m = res["metrics"]
    # ball 0-20 s looked at (minus 2 s background), block 20-40 not looked at, car 40-55 looked at
    assert 55 < m["looking_at_played_pct"] < 70
    for f in res["files"].values():
        assert f is None or Path(f).exists()


def test_auto_discovery():
    res = run(objects=(), out=ROOT / "tests" / "out_auto", make_review_video=False)
    assert set(res["objects"]) == {"ball", "block", "toy car"}, res["objects"]


if __name__ == "__main__":
    r = run()
    a = accuracy(r)
    print("sync:", r["sync"])
    print("metrics:", r["metrics"])
    print(f"interaction accuracy {a[0]:.3f}, gaze accuracy {a[1]:.3f} over {a[2]} samples")
    print("files:", r["files"])
    print("timing:", r["timing"])
    print("warnings:", r["warnings"])
    test_auto_discovery()
    print("auto-discovery ok")
