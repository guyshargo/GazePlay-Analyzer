"""Start GazePlay Analyzer: opens the interface in your web browser.

    python app.py              (default port 8765)
    python app.py --port 9000 --no-browser
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gazeplay.config import APP_NAME, APP_VERSION  # noqa: E402
from gazeplay.web.server import app, cleanup_old_uploads  # noqa: E402


def free_port(preferred: int) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


def main():
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    cleanup_old_uploads()
    print(f"\n  {APP_NAME} {APP_VERSION}\n  Open {url} in your browser (keep this window open).\n"
          f"  Press Ctrl+C to quit.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
