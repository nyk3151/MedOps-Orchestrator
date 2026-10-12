#!/bin/bash
# Usage: entrypoint.sh <absolute path to config.json>   (or MEDOPS_CONFIG=<path>)
# Slicer needs an X display even headless -> Xvfb. Exit code comes from run_pipeline.py.
set -euo pipefail
CONFIG="${1:-${MEDOPS_CONFIG:-}}"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
    echo "config.json not found: '${CONFIG}'" >&2
    exit 2
fi
# Hard limit: if the script dies before slicer.util.exit(), Slicer would never exit.
exec timeout --kill-after=60 "${MEDOPS_TIMEOUT_S:-3600}" \
    xvfb-run -a -s "-screen 0 1280x1024x24" \
    /opt/slicer/Slicer --no-splash --no-main-window \
    --python-script /app/run_pipeline.py "$CONFIG"
