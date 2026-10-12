"""Start the job server and the Streamlit UI together (Ctrl+C stops both).

    python -m medops.launch            # MOCK analysis (default)
    python -m medops.launch --live     # real 3D Slicer analysis (validated before start)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from medops.settings import load_settings

APP = Path(__file__).resolve().parent / "ui" / "app.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="run real 3D Slicer analyses")
    ap.add_argument("--ui-port", type=int, default=8501)
    args = ap.parse_args()

    settings = load_settings(live=True if args.live else None)
    if settings.analysis_live:
        settings.validate_live_analysis()
    url = urlparse(settings.job_server_url)
    server_cmd = [sys.executable, "-m", "medops.analysis.server",
                  "--host", url.hostname or "127.0.0.1", "--port", str(url.port or 8000)]
    if args.live:
        server_cmd.append("--live")

    procs = [subprocess.Popen(server_cmd)]
    try:
        for _ in range(60):
            try:
                requests.get(f"{settings.job_server_url}/health", timeout=1)
                break
            except requests.RequestException:
                if procs[0].poll() is not None:
                    print("job server failed to start", file=sys.stderr)
                    return 1
                time.sleep(0.5)
        procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(APP),
                                       "--server.port", str(args.ui_port),
                                       "--server.headless", "true"]))
        print(f"\nMed-Ops Orchestrator: http://localhost:{args.ui_port}  (Ctrl+C to stop)\n", flush=True)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
