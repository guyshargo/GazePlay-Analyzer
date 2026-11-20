#!/usr/bin/env bash
# Runs the synthetic end-to-end test (no YOLO models needed).
cd "$(dirname "$0")"
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" tests/test_synthetic.py
