"""Central settings resolution.

Resolution order per key (fixed):  environment variable > settings.yaml > repo-relative default

- Defaults are repo-relative (./data/...) so a fresh clone runs in MOCK mode with zero config.
- Relative paths in settings.yaml are resolved against the repository root, never the cwd
  (Streamlit, uvicorn and Slicer all start with different working directories).
- LIVE analysis must be requested explicitly (``--live`` on the job server or
  ``MEDOPS_ANALYSIS_LIVE=1``). It is deliberately NOT settable from settings.yaml so a
  forgotten config file can never start a 13-minute GPU run.
- Anything LIVE validates up-front (``validate_*``) and fails with the full list of what is
  missing, instead of hitting FileNotFoundError ten minutes into a job.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class SettingsError(RuntimeError):
    """Raised when a LIVE component is requested but its configuration is incomplete."""

    def __init__(self, component: str, problems: list[str]):
        self.component = component
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(
            f"{component} の設定が不足しています（環境変数 > settings.yaml > 既定 の順で解決）:\n{lines}"
        )


# key -> (environment variable, default). Path defaults are repo-relative strings.
_SPEC: dict[str, tuple[str, Any]] = {
    "data_dir": ("MEDOPS_DATA_DIR", "data"),
    "dicom_dir": ("MEDOPS_DICOM_DIR", None),      # default: {data_dir}/dicom
    "nifti_dir": ("MEDOPS_NIFTI_DIR", None),      # default: {data_dir}/nifti
    "jobs_dir": ("MEDOPS_JOBS_DIR", None),        # default: {data_dir}/jobs
    "results_dir": ("MEDOPS_RESULTS_DIR", None),  # default: {data_dir}/results
    # --- analysis engine (LIVE only) ---
    # local   : host Slicer.exe            (slicer_exe, module_dir, aorta_model, tl_model)
    # docker  : GPU worker container here  (worker_image, aorta_model, tl_model)
    # cloudrun: Cloud Run Job on an L4 GPU (gpu_job, run_region; aorta_model / tl_model are
    #           paths inside the GCS mount shared by the job server and the GPU job)
    "engine_backend": ("MEDOPS_ENGINE_BACKEND", "local"),
    "worker_image": ("MEDOPS_WORKER_IMAGE", None),
    "gpu_job": ("MEDOPS_GPU_JOB", None),
    "run_region": ("MEDOPS_RUN_REGION", "asia-southeast1"),
    "module_version": ("MEDOPS_MODULE_VERSION", None),  # override (container backends)
    # Cost guard for public demos: max LIVE analyses accepted per UTC day (unset = no limit)
    "max_live_jobs_per_day": ("MEDOPS_MAX_LIVE_JOBS_PER_DAY", None),
    "slicer_exe": ("MEDOPS_SLICER_EXE", None),
    "module_dir": ("MEDOPS_MODULE_DIR", None),
    "aorta_model": ("MEDOPS_AORTA_MODEL", None),
    "tl_model": ("MEDOPS_TL_MODEL", None),
    "pipeline_script": ("MEDOPS_PIPELINE_SCRIPT", "scripts/run_pipeline.py"),
    "job_server_url": ("MEDOPS_JOB_SERVER_URL", "http://127.0.0.1:8000"),
    "mock_delay_s": ("MEDOPS_MOCK_DELAY_S", 5.0),  # MOCK analysis duration (demo progress)
    # --- PACS ---
    "pacs_mode": ("MEDOPS_PACS_MODE", "mock"),    # mock | live
    "orthanc_url": ("MEDOPS_ORTHANC_URL", "http://127.0.0.1:8042"),
    "orthanc_user": ("MEDOPS_ORTHANC_USER", None),
    "orthanc_password": ("MEDOPS_ORTHANC_PASSWORD", None),
    # Browser-facing PACS URL for the viewer links (default: orthanc_url)
    "pacs_viewer_url": ("MEDOPS_PACS_VIEWER_URL", None),
    # When data_dir is a GCS mount: bucket + object prefix of data_dir, so large files
    # (Slicer scenes) can be handed out as short-lived signed links instead of proxied
    "artifact_bucket": ("MEDOPS_ARTIFACT_BUCKET", None),
    "artifact_prefix": ("MEDOPS_ARTIFACT_PREFIX", "data"),
    # --- DB --- (default: local SQLite file, so the DB layer works without Docker)
    "database_url": ("MEDOPS_DATABASE_URL", None),
    # --- LLM ---
    "llm_mode": ("MEDOPS_LLM_MODE", "mock"),      # mock | gemini
    "gcp_project": ("GOOGLE_CLOUD_PROJECT", None),
    "gcp_location": ("GOOGLE_CLOUD_LOCATION", "asia-northeast1"),
    "gcp_credentials": ("GOOGLE_APPLICATION_CREDENTIALS", None),
    "gemini_model": ("MEDOPS_GEMINI_MODEL", "gemini-2.5-flash"),
}

_PATH_KEYS = {
    "data_dir", "dicom_dir", "nifti_dir", "jobs_dir", "results_dir",
    "slicer_exe", "module_dir", "aorta_model", "tl_model", "pipeline_script", "gcp_credentials",
}

_TRUTHY = {"1", "true", "yes", "on"}


def on_cloud_run() -> bool:
    """Cloud Run sets K_SERVICE (services) / CLOUD_RUN_JOB (jobs). There the attached runtime
    service account is used instead of a key file (keys never go into images)."""
    return bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB"))


def _settings_yaml_path() -> Path:
    override = os.environ.get("MEDOPS_SETTINGS")
    return Path(override) if override else REPO_ROOT / "settings.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SettingsError("settings.yaml", [f"{path} の最上位はマッピングである必要があります"])
    unknown = sorted(set(data) - set(_SPEC))
    if unknown:
        raise SettingsError("settings.yaml", [f"未知のキー: {', '.join(unknown)}"])
    return data


def _to_path(value: Optional[str]) -> Optional[Path]:
    if value in (None, ""):
        return None
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    dicom_dir: Path
    nifti_dir: Path
    jobs_dir: Path
    results_dir: Path
    slicer_exe: Optional[Path]
    module_dir: Optional[Path]
    aorta_model: Optional[Path]
    tl_model: Optional[Path]
    pipeline_script: Path
    engine_backend: str
    worker_image: Optional[str]
    gpu_job: Optional[str]
    run_region: str
    module_version: Optional[str]
    max_live_jobs_per_day: Optional[int]
    job_server_url: str
    mock_delay_s: float
    pacs_mode: str
    orthanc_url: str
    orthanc_user: Optional[str]
    orthanc_password: Optional[str]
    pacs_viewer_url: str
    artifact_bucket: Optional[str]
    artifact_prefix: str
    database_url: str
    llm_mode: str
    gcp_project: Optional[str]
    gcp_location: str
    gcp_credentials: Optional[Path]
    gemini_model: str
    analysis_live: bool
    sources: dict  # key -> "env" | "yaml" | "default" (for diagnostics; values never logged)

    # ---------------- validation (call at startup of LIVE components) ----------------

    def validate_live_analysis(self) -> None:
        problems = []

        def need_file(key: str, value: Optional[Path]) -> None:
            if value is None:
                problems.append(f"{key} が未設定です（{_SPEC[key][0]} または settings.yaml）")
            elif not value.is_file():
                problems.append(f"{key} のファイルが存在しません: {value}")

        def need_dir(key: str, value: Optional[Path]) -> None:
            if value is None:
                problems.append(f"{key} が未設定です（{_SPEC[key][0]} または settings.yaml）")
            elif not value.is_dir():
                problems.append(f"{key} のディレクトリが存在しません: {value}")

        need_dir("aorta_model", self.aorta_model)
        need_dir("tl_model", self.tl_model)
        if self.engine_backend == "local":
            need_file("slicer_exe", self.slicer_exe)
            need_dir("module_dir", self.module_dir)
            if self.module_dir is not None and self.module_dir.is_dir() \
                    and not (self.module_dir / "AorticPipeline.py").is_file():
                problems.append(f"module_dir に AorticPipeline.py がありません: {self.module_dir}")
            need_file("pipeline_script", self.pipeline_script)
        elif self.engine_backend == "docker":
            if not self.worker_image:
                problems.append("worker_image が未設定です（MEDOPS_WORKER_IMAGE または settings.yaml）")
        elif self.engine_backend == "cloudrun":
            if not self.gpu_job:
                problems.append("gpu_job（Cloud Run Job 名）が未設定です（MEDOPS_GPU_JOB）")
            if not self.gcp_project:
                problems.append("GOOGLE_CLOUD_PROJECT が未設定です")
        if problems:
            raise SettingsError("LIVE 解析エンジン", problems)

    def validate_live_pacs(self) -> None:
        problems = []
        if self.pacs_mode != "live":
            problems.append(f"pacs_mode が live ではありません（現在: {self.pacs_mode}）")
        if not self.orthanc_url:
            problems.append("orthanc_url が未設定です")
        if problems:
            raise SettingsError("Orthanc (PACS)", problems)

    def validate_gemini(self) -> None:
        problems = []
        if not self.gcp_project:
            problems.append("GOOGLE_CLOUD_PROJECT が未設定です")
        if not self.gcp_location:
            problems.append("GOOGLE_CLOUD_LOCATION が未設定です")
        if self.gcp_credentials is None:
            if not on_cloud_run():
                problems.append("GOOGLE_APPLICATION_CREDENTIALS（サービスアカウント鍵）が未設定です")
        elif not self.gcp_credentials.is_file():
            problems.append(f"サービスアカウント鍵が存在しません: {self.gcp_credentials}")
        if problems:
            raise SettingsError("Gemini (Vertex AI)", problems)

    def summary(self) -> dict:
        """Non-secret overview for logs / UI (no passwords, no credential paths)."""
        return {
            "analysis": "LIVE" if self.analysis_live else "MOCK",
            "pacs": self.pacs_mode.upper(),
            "llm": self.llm_mode.upper(),
            "db": self.database_url.split(":", 1)[0],
        }


def load_settings(*, live: Optional[bool] = None) -> Settings:
    """Resolve settings. ``live`` forces LIVE analysis (job server ``--live`` flag)."""
    yaml_values = _load_yaml(_settings_yaml_path())
    raw: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key, (env, default) in _SPEC.items():
        if os.environ.get(env) not in (None, ""):
            raw[key], sources[key] = os.environ[env], "env"
        elif yaml_values.get(key) not in (None, ""):
            raw[key], sources[key] = yaml_values[key], "yaml"
        else:
            raw[key], sources[key] = default, "default"

    paths = {k: _to_path(raw[k]) for k in _PATH_KEYS}
    data_dir = paths["data_dir"]
    for sub in ("dicom", "nifti", "jobs", "results"):
        key = f"{sub}_dir"
        if paths[key] is None:
            paths[key] = data_dir / sub

    database_url = raw["database_url"] or f"sqlite:///{(data_dir / 'medops.db').as_posix()}"

    engine_backend = str(raw["engine_backend"]).lower()
    pacs_mode = str(raw["pacs_mode"]).lower()
    llm_mode = str(raw["llm_mode"]).lower()
    problems = []
    if engine_backend not in {"local", "docker", "cloudrun"}:
        problems.append(f"engine_backend は local / docker / cloudrun のいずれか（現在: {engine_backend}）")
    if pacs_mode not in {"mock", "live"}:
        problems.append(f"pacs_mode は mock / live のいずれか（現在: {pacs_mode}）")
    if llm_mode not in {"mock", "gemini"}:
        problems.append(f"llm_mode は mock / gemini のいずれか（現在: {llm_mode}）")
    if problems:
        raise SettingsError("settings", problems)

    if live is None:
        live = os.environ.get("MEDOPS_ANALYSIS_LIVE", "").lower() in _TRUTHY

    return Settings(
        **{k: paths[k] for k in _PATH_KEYS},
        engine_backend=engine_backend,
        worker_image=raw["worker_image"],
        gpu_job=raw["gpu_job"],
        run_region=str(raw["run_region"]),
        module_version=raw["module_version"],
        max_live_jobs_per_day=int(raw["max_live_jobs_per_day"]) if raw["max_live_jobs_per_day"] else None,
        job_server_url=str(raw["job_server_url"]).rstrip("/"),
        mock_delay_s=float(raw["mock_delay_s"]),
        pacs_mode=pacs_mode,
        orthanc_url=str(raw["orthanc_url"]).rstrip("/"),
        orthanc_user=raw["orthanc_user"],
        orthanc_password=raw["orthanc_password"],
        pacs_viewer_url=str(raw["pacs_viewer_url"] or raw["orthanc_url"]).rstrip("/"),
        artifact_bucket=raw["artifact_bucket"],
        artifact_prefix=str(raw["artifact_prefix"]).strip("/"),
        database_url=str(database_url),
        llm_mode=llm_mode,
        gcp_project=raw["gcp_project"],
        gcp_location=str(raw["gcp_location"]),
        gemini_model=str(raw["gemini_model"]),
        analysis_live=bool(live),
        sources=sources,
    )
