#!/usr/bin/env bash
# GazePlay Analyzer - first run installs everything, later runs just start the app.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

if [ ! -f ".venv/installed.ok" ]; then
  echo
  echo "  First-time setup: installing GazePlay Analyzer (5-15 minutes, needs internet)."
  echo
  if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    echo "  Python 3.10 or newer is needed: https://www.python.org/downloads/"
    exit 1
  fi
  rm -rf .venv
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  if [ "$(uname)" = "Darwin" ]; then
    .venv/bin/python -m pip install torch torchvision
  else
    .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python setup_models.py
  touch .venv/installed.ok
fi

exec .venv/bin/python app.py "$@"
