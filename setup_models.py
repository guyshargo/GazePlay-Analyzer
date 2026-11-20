"""Download the detection models once, so later analyses also work offline.

Run automatically by the setup scripts; safe to run again.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gazeplay.config import MODELS_DIR, Settings  # noqa: E402
from gazeplay.detectors import UltralyticsBackend  # noqa: E402


def main():
    print(f"Models folder: {MODELS_DIR}")
    s = Settings.from_preset("balanced")
    be = UltralyticsBackend(log=print)
    be.configure(s)
    print("1/3 Object detector + text encoder (YOLOE)...")
    be.prepare_objects(["ball", "teddy bear", "person"])
    img = np.zeros((480, 640, 3), np.uint8)
    be.detect_objects(img, 640)
    print("2/3 Person / pose model...")
    be.detect_people(img, 640)
    print("3/3 Prompt-free model for automatic object discovery...")
    be.discover_objects([img], 640)
    print(f"\nAll models ready (device: {be.device}).")


if __name__ == "__main__":
    main()
