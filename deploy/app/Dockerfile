# Med-Ops app image: one image, two Cloud Run services (command differs).
#   API (job server): python -m medops.analysis.server --host 0.0.0.0 --port 8080 [--live]
#   UI  (Streamlit) : streamlit run /app/src/medops/ui/app.py --server.port 8080 ...
# No keys and no data in the image: config comes from env vars / Secret Manager.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MEDOPS_DATA_DIR=/tmp/medops-data

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install .

RUN useradd --create-home --uid 10001 medops
USER medops

EXPOSE 8080
CMD ["python", "-m", "medops.analysis.server", "--host", "0.0.0.0", "--port", "8080"]
