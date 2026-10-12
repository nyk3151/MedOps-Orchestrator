"""Analysis engine adapter: builds the per-job config.json and runs (or mocks) 3D Slicer.

Ported from the previous project's job_server.py. The Slicer side lives in
scripts/run_pipeline.py and runs in Slicer's bundled Python; this module only talks to it
through config.json (in) and results.json / run.log (out) inside data/jobs/{job_id}/.

Backends (settings.engine_backend), all running the SAME scripts/run_pipeline.py:
  local    : host Slicer.exe (verified on Windows)
  docker   : the Linux GPU worker image (deploy/gpu) on this machine, --gpus all
  cloudrun : one Cloud Run Job execution on an L4 GPU per analysis (scale to zero). The job
             server and the job mount the same GCS bucket at the same path.
config.json always holds the paths AS SEEN BY THE WORKER.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from medops.settings import Settings

MODULE_NAME = "AorticPipeline"
DEFAULT_FOLDS = "0,1,2,3,4"
DEFAULT_ROI_SIZE = 100.0
DEFAULT_CPR_SPACING = 0.5
# Same as the module's GUI defaults (ROI 100 mm, CPR slice 100 mm). runPipeline()'s own
# keyword default is cprSliceSize=40: a 100 mm ROI on a 40 mm straightened section starts at
# a negative index and measures only a 30 mm corner (found on 001, 2026-10-13).
DEFAULT_CPR_SLICE_SIZE = 100.0
DEFAULT_RUN_TL = True
MOCK_MODULE_VERSION = "mock"
MOCK_SLICER_VERSION = "mock"

# Paths inside the GPU worker image / container (deploy/gpu/Dockerfile).
WORKER_DATA = "/data"
WORKER_AORTA_MODEL = "/models/aorta"
WORKER_TL_MODEL = "/models/tl"
WORKER_MODULE_DIR = "/opt/AorticPipeline/AorticPipeline"
DOCKER_SHM = "8g"            # nnU-Net worker processes share tensors via /dev/shm
CLOUDRUN_POLL_S = 20


class AnalysisError(RuntimeError):
    pass


@dataclass
class EngineResult:
    rows: list[dict]                       # normalized: tl_ratio_pct (not "tl_ratio_%")
    slicer_version: Optional[str] = None
    raw: dict = field(default_factory=dict)


def nifti_path_for(settings: Settings, case_id: str) -> Path:
    """{nifti_dir}/{case_id}_0000.nii.gz — same naming as the verified analysis inputs."""
    return settings.nifti_dir / f"{case_id}_0000.nii.gz"


def _in_docker_worker(settings: Settings, host_path: Path) -> str:
    try:
        rel = host_path.resolve().relative_to(settings.data_dir.resolve())
    except ValueError as exc:
        raise AnalysisError(f"docker backend: {host_path.name} must live under data_dir") from exc
    return f"{WORKER_DATA}/{rel.as_posix()}"


def build_config(settings: Settings, job_dir: Path, case_id: str, ct_path: Path,
                 levels: list[str]) -> dict:
    """All paths absolute, as seen by the worker (Slicer's cwd is its install directory)."""
    backend = settings.engine_backend
    if backend == "docker":
        paths = {"input": _in_docker_worker(settings, ct_path),
                 "output_dir": _in_docker_worker(settings, job_dir),
                 "module_dir": WORKER_MODULE_DIR,
                 "aorta_model": WORKER_AORTA_MODEL, "tl_model": WORKER_TL_MODEL}
    elif backend == "cloudrun":  # same GCS mount path in the job server and the GPU job
        paths = {"input": ct_path.as_posix(), "output_dir": job_dir.as_posix(),
                 "module_dir": WORKER_MODULE_DIR,
                 "aorta_model": settings.aorta_model.as_posix() if settings.aorta_model else None,
                 "tl_model": settings.tl_model.as_posix() if settings.tl_model else None}
    else:
        paths = {"input": str(ct_path.resolve()), "output_dir": str(job_dir.resolve()),
                 "module_dir": str(settings.module_dir) if settings.module_dir else None,
                 "aorta_model": str(settings.aorta_model) if settings.aorta_model else None,
                 "tl_model": str(settings.tl_model) if settings.tl_model else None}
    return {
        "input": paths["input"],
        "case_id": case_id,
        "output_dir": paths["output_dir"],
        "module_dir": paths["module_dir"],
        "aorta_model": paths["aorta_model"],
        "tl_model": paths["tl_model"],
        "aorta_folds": DEFAULT_FOLDS,
        "tl_folds": DEFAULT_FOLDS,
        "roi_size": DEFAULT_ROI_SIZE,
        "cpr_spacing": DEFAULT_CPR_SPACING,
        "cpr_slice_size": DEFAULT_CPR_SLICE_SIZE,
        "run_tl": DEFAULT_RUN_TL,
        "vertebra_codes": levels,
    }


def model_name(path: Optional[Path]) -> Optional[str]:
    """Identifier for the weights (directory name only — never the full path)."""
    return path.name if path else None


def module_version(settings: Settings) -> str:
    """git describe of the AorticPipeline checkout, else a content hash of AorticPipeline.py.

    Container backends: the version baked into the worker image (label / deploy setting)."""
    if not settings.analysis_live:
        return MOCK_MODULE_VERSION
    if settings.module_version:
        return settings.module_version
    if settings.engine_backend == "docker" and settings.worker_image:
        try:
            out = subprocess.run(["docker", "image", "inspect", "--format",
                                  '{{index .Config.Labels "medops.module_version"}}',
                                  settings.worker_image], capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return "unknown"
    if settings.module_dir is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(settings.module_dir), "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    src = settings.module_dir / "AorticPipeline.py"
    return "sha256:" + hashlib.sha256(src.read_bytes()).hexdigest()[:12]


_THORACIC_DISPLAY = re.compile(r"^Th(\d{1,2})$", re.IGNORECASE)


def canonical_level(label: str) -> str:
    """AorticPipeline labels thoracic levels for display as 'Th8'; the project uses 'T8'."""
    m = _THORACIC_DISPLAY.match(label.strip())
    return f"T{m.group(1)}" if m else label.strip().upper()


def normalize_rows(rows: Optional[list[dict]]) -> list[dict]:
    """run_pipeline.py emits 'tl_ratio_%' and 'Th8'; the DB/API use 'tl_ratio_pct' and 'T8'."""
    out = []
    for r in rows or []:
        out.append({
            "level": canonical_level(r["level"]),
            "aorta_area_mm2": r.get("aorta_area_mm2"),
            "aorta_dia_mm": r.get("aorta_dia_mm"),
            "tl_area_mm2": r.get("tl_area_mm2"),
            "fl_area_mm2": r.get("fl_area_mm2"),
            "tl_ratio_pct": r.get("tl_ratio_pct", r.get("tl_ratio_%")),
        })
    return out


def mock_results(case_id: str, levels: list[str], run_tl: bool = DEFAULT_RUN_TL) -> dict:
    """Deterministic per case_id (same as the original MOCK job server)."""
    rng = random.Random(case_id)
    rows = []
    for level in levels:
        aorta = round(rng.uniform(300.0, 700.0), 2)
        rec = {
            "level": level,
            "aorta_area_mm2": aorta,
            "aorta_dia_mm": round(2 * math.sqrt(aorta / math.pi), 2),
        }
        if run_tl:
            tl = round(aorta * rng.uniform(0.4, 0.8), 2)
            rec["tl_area_mm2"] = tl
            rec["fl_area_mm2"] = round(aorta - tl, 2)
            rec["tl_ratio_%"] = round(tl / aorta * 100, 1)
        rows.append(rec)
    return {"status": "success", "case_id": case_id, "results": rows,
            "slicer_version": MOCK_SLICER_VERSION}


async def run_engine(settings: Settings, job_dir: Path, config: dict,
                     mock_delay_s: float = 0.0) -> EngineResult:
    """Run one analysis. config.json must already be written to job_dir."""
    result_path = job_dir / "results.json"

    if not settings.analysis_live:
        if mock_delay_s:
            await asyncio.sleep(mock_delay_s)
        data = mock_results(config["case_id"], config["vertebra_codes"], config["run_tl"])
        result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        from medops.analysis import views  # local import: views uses canonical_level from here
        views.write_synthetic(job_dir, data["results"], config["case_id"])
        return EngineResult(rows=normalize_rows(data["results"]),
                            slicer_version=data["slicer_version"], raw=data)

    if settings.engine_backend == "cloudrun":
        detail = await asyncio.to_thread(_run_cloudrun_job, settings, config)
        # Results written by the job through the GCS mount can show up here with a delay.
        for _ in range(12):
            if result_path.exists():
                break
            await asyncio.sleep(5)
    else:
        if settings.engine_backend == "docker":
            cmd = _docker_cmd(settings, job_dir, config)
        else:
            cmd = [str(settings.slicer_exe), "--no-splash", "--no-main-window",
                   "--python-script", str(settings.pipeline_script),
                   str((job_dir / "config.json").resolve())]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        detail = f"returncode={proc.returncode}. stderr: {stderr.decode(errors='replace')[-2000:]}"

    if not result_path.exists():
        raise AnalysisError(f"results.json not produced ({detail}). See {job_dir.name}/run.log")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    if data.get("status") != "success":
        raise AnalysisError(data.get("trace", "unknown error"))
    if not data.get("results"):
        # run_pipeline.py writes status=success with results=None when FinalResults is missing.
        raise AnalysisError(f"FinalResults table not found / empty. See {job_dir.name}/run.log")
    return EngineResult(rows=normalize_rows(data["results"]),
                        slicer_version=data.get("slicer_version"), raw=data)


def _docker_cmd(settings: Settings, job_dir: Path, config: dict) -> list[str]:
    def bind(src: Path, dst: str, ro: bool = False) -> list[str]:
        opt = f"type=bind,source={src.resolve()},target={dst}" + (",readonly" if ro else "")
        return ["--mount", opt]

    return ["docker", "run", "--rm", "--gpus", "all", "--shm-size", DOCKER_SHM,
            "--name", f"medops-job-{job_dir.name[:12]}",
            *bind(settings.data_dir, WORKER_DATA),
            *bind(settings.aorta_model, WORKER_AORTA_MODEL, ro=True),
            *bind(settings.tl_model, WORKER_TL_MODEL, ro=True),
            settings.worker_image, f"{config['output_dir']}/config.json"]


def _authorized_session():
    """Credentials of the attached Cloud Run runtime service account (metadata server)."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds)


def _run_cloudrun_job(settings: Settings, config: dict) -> str:
    """Execute the GPU Cloud Run Job for one config and block until it finishes.

    Runs inside the job-server service on Cloud Run, authenticated as its attached runtime
    service account (metadata server) - no key file involved.
    """
    import time

    session = _authorized_session()
    base = "https://run.googleapis.com/v2"
    job = f"projects/{settings.gcp_project}/locations/{settings.run_region}/jobs/{settings.gpu_job}"
    r = session.post(f"{base}/{job}:run", json={"overrides": {
        "containerOverrides": [{"args": [f"{config['output_dir']}/config.json"]}],
        "taskCount": 1}}, timeout=60)
    if r.status_code >= 400:
        raise AnalysisError(f"Cloud Run Job の起動に失敗しました: HTTP {r.status_code} {r.text[:500]}")
    op = r.json()
    while not op.get("done"):
        time.sleep(CLOUDRUN_POLL_S)
        op = session.get(f"{base}/{op['name']}", timeout=60).json()
    # Like the local backends, the exit status is only a diagnostic: results.json decides.
    # (Linux Slicer can crash in its own shutdown after run_pipeline.py has written a
    # complete results.json; the caller raises if that file is missing or not a success.)
    if "error" in op:
        return f"Cloud Run Job が失敗しました: {op['error'].get('message', op['error'])}"
    execution = op.get("response", {})
    name = execution.get("name", "?").rsplit("/", 1)[-1]
    if execution.get("failedCount") or not execution.get("succeededCount"):
        return (f"execution={name} failed (succeeded={execution.get('succeededCount')}, "
                f"failed={execution.get('failedCount')})")
    return f"execution={name}"
