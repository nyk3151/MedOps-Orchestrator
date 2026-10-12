# GPU analysis worker: 3D Slicer 5.8.1 (Linux) + AorticPipeline, run headless under Xvfb.
# Runs scripts/run_pipeline.py exactly like the Windows host does (one positional config.json).
#
# Build (the AorticPipeline checkout is passed as a named context; it is NOT copied into
# this repository, and model weights are NOT baked into the image — they are mounted):
#
#   git -C <AorticPipeline checkout> archive --format=tar <commit> | tar -x -C <tmpdir>
#   docker build -f deploy/gpu/Dockerfile --build-context aorticpipeline=<tmpdir> \
#       --build-arg MODULE_VERSION=<git describe> -t medops-gpu-worker .
#
# Run (local GPU):
#   docker run --rm --gpus all -v <data>:/data -v <aorta_model>:/models/aorta:ro \
#       -v <tl_model>:/models/tl:ro medops-gpu-worker /data/jobs/<job_id>/config.json
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/root \
    TOTALSEG_HOME_DIR=/opt/totalseg \
    QT_QPA_PLATFORM=xcb

# Runtime libraries for Slicer's Qt/VTK + Xvfb (no GPU driver: the NVIDIA runtime injects it).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xvfb xauth \
        libgl1 libglu1-mesa libegl1 libopengl0 libglib2.0-0 libdbus-1-3 libnss3 \
        libfontconfig1 libfreetype6 libxrender1 libxi6 libxrandr2 libxt6 libsm6 libice6 \
        libxcomposite1 libxdamage1 libxtst6 libxkbcommon0 libxkbcommon-x11-0 libxkbfile1 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxcb-cursor0 libxcb-xinput0 \
        libasound2 libpulse0 libpulse-mainloop-glib0 libxcursor1 libcups2 libhwloc15 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3D Slicer 5.8.1, revision 33241 (same build as the verified Windows install). Pinned by sha512.
ARG SLICER_URL=https://download.slicer.org/bitstream/67c51fc129825655577cfee9
ARG SLICER_SHA512=56a3b278b72124344c6ddd0be4a27f467c996032ae477226b4a9f7a3c64eec4047aaf8d8868580854d4acfdb00679118b3fbab4ecf955e838ce010338036c9c8
RUN curl -fsSL "$SLICER_URL" -o /tmp/slicer.tar.gz \
    && echo "$SLICER_SHA512  /tmp/slicer.tar.gz" | sha512sum -c - \
    && mkdir -p /opt/slicer \
    && tar -xzf /tmp/slicer.tar.gz -C /opt/slicer --strip-components=1 \
    && rm /tmp/slicer.tar.gz

# Extensions + Python packages pinned to the versions of the verified Windows install,
# and the open TotalSegmentator 'total' fast (3 mm) weights used for vertebra levels.
COPY deploy/gpu/constraints.txt /tmp/constraints.txt
COPY deploy/gpu/setup_slicer.py /tmp/setup_slicer.py
# Two Slicer sessions: extension modules are only loaded after a restart (see setup_slicer.py).
# timeout: an error before slicer.util.exit() would otherwise leave Slicer running forever.
RUN timeout 3600 xvfb-run -a /opt/slicer/Slicer --no-splash --no-main-window \
        --python-script /tmp/setup_slicer.py phase1 \
    && timeout 3600 xvfb-run -a /opt/slicer/Slicer --no-splash --no-main-window \
        --python-script /tmp/setup_slicer.py phase2 \
    ; cat /opt/medops-setup.log && test -f /opt/medops-setup.ok

COPY --from=aorticpipeline . /opt/AorticPipeline
COPY scripts/run_pipeline.py /app/run_pipeline.py
COPY deploy/gpu/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ARG MODULE_VERSION=unknown
LABEL org.opencontainers.image.title="medops-gpu-worker" \
      medops.module_version="$MODULE_VERSION" \
      medops.slicer_version="5.8.1-r33241"
ENV MEDOPS_MODULE_VERSION=$MODULE_VERSION

# Cloud Run GPU mounts the NVIDIA driver (580 / CUDA 13) here; torch cu121 runs on it.
# UTF-8 locale: Slicer reads --python-script with the locale encoding, and the verified
# run_pipeline.py contains non-ASCII comments (it does not even parse under "C"/ASCII).
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1

ENTRYPOINT ["/app/entrypoint.sh"]
