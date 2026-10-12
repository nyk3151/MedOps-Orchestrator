"""FastAPI job server: one analysis at a time (shared GPU), never synchronous.

Ported from the previous project's job_server.py. Callers only send case_id (+ levels);
every path is resolved here from settings, so no filesystem path ever comes from an API
caller or the LLM.

Safe by default: MOCK mode (no Slicer, deterministic dummy results) unless LIVE is
explicitly requested. LIVE validates its configuration at startup and refuses to start
when anything is missing.

    python -m medops.analysis.server            # MOCK
    python -m medops.analysis.server --live     # LIVE (or MEDOPS_ANALYSIS_LIVE=1 with uvicorn)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from medops.analysis import engine, views
from medops.db.repo import Repository
from medops.ids import normalize_levels, validate_case_id
from medops.pacs.orthanc import CaseNotFoundError, PacsError
from medops.pacs.service import get_pacs
from medops.settings import Settings, load_settings

log = logging.getLogger("medops.analysis")

JobStatus = Literal["queued", "running", "success", "error"]
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")   # uuid4().hex; job ids address directories
SCENE_FILE = "scene.mrb"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:  # SQLite drops tzinfo; we only ever store UTC
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


class JobCreateRequest(BaseModel):
    case_id: str
    levels: Optional[list[str]] = None
    study_instance_uid: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    queue_position: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    case_id: str
    levels: list[str]
    study_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None
    study_date: Optional[str] = None
    modality: Optional[str] = None
    stage: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    elapsed_s: Optional[float] = None
    results: Optional[list[dict]] = None
    error: Optional[str] = None
    analysis_mode: Optional[str] = None
    run_id: Optional[int] = None


class DailyLimitError(RuntimeError):
    pass


class JobRunner:
    """In-process FIFO queue with a single worker (GPU concurrency = 1)."""

    def __init__(self, settings: Settings, pacs=None, repo: Optional[Repository] = None):
        self.settings = settings
        self.pacs = pacs if pacs is not None else get_pacs(settings)
        self.repo = repo if repo is not None else Repository.from_settings(settings)
        self.jobs: dict[str, dict] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.mode = "live" if settings.analysis_live else "mock"
        self.module_version = engine.module_version(settings)

    # -- public API -----------------------------------------------------------------

    def submit(self, case_id: str, levels: list[str],
               study_instance_uid: Optional[str] = None) -> dict:
        cap = self.settings.max_live_jobs_per_day
        if self.mode == "live" and cap:
            midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            if self.repo.count_runs_since(midnight, "live") >= cap:
                raise DailyLimitError(f"本日の解析上限（{cap} 件）に達しました。明日（UTC）以降に再度お試しください。")
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id, "status": "queued", "case_id": case_id, "levels": levels,
            "study_instance_uid": study_instance_uid, "series_instance_uid": None,
            "study_date": None, "modality": None, "stage": "queued", "message": "キュー待ち", "created_at": now_iso(),
            "started_at": None, "finished_at": None, "elapsed_s": None,
            "results": None, "error": None, "analysis_mode": self.mode,
        }
        # DB first: if the DB is down the job is rejected instead of running untracked.
        job["run_id"] = self.repo.create_run(
            job_id=job_id, case_id=case_id, levels=levels, analysis_mode=self.mode,
            module_name=engine.MODULE_NAME, created_at=job["created_at"])
        self.jobs[job_id] = job
        return job  # caller enqueues on the event loop (asyncio.Queue is not thread-safe)

    def get(self, job_id: str) -> Optional[dict]:
        """Live in-memory state if this process ran it, else the DB record (history)."""
        if job_id in self.jobs:
            return self.jobs[job_id]
        run = self.repo.get_run(job_id)
        if run is None:
            return None
        return {
            "job_id": run["job_id"], "status": run["status"], "case_id": run["case_id"],
            "levels": run["levels"].split(","), "stage": run["stage"],
            "study_instance_uid": run["study_instance_uid"],
            "series_instance_uid": run["series_instance_uid"],
            "study_date": run["study_date"], "modality": run["modality"],
            "created_at": _iso(run["created_at"]), "started_at": _iso(run["started_at"]),
            "finished_at": _iso(run["finished_at"]), "elapsed_s": run["elapsed_s"],
            "results": run["results"], "error": run["error_message"],
            "analysis_mode": run["analysis_mode"], "run_id": run["run_id"],
        }

    def pending_ahead(self, job_id: str) -> int:
        return sum(1 for j in self.jobs.values()
                   if j["status"] in ("queued", "running") and j["job_id"] != job_id
                   and j["created_at"] <= self.jobs[job_id]["created_at"])

    async def worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs[job_id]
            job["status"] = "running"
            job["started_at"] = now_iso()
            t0 = asyncio.get_running_loop().time()
            try:
                await asyncio.to_thread(self.repo.update_run, job_id, status="running",
                                        started_at=job["started_at"])
                await self._run(job)
                job["status"] = "success"
                job["stage"] = "done"
                job["message"] = "完了（DB登録済み）"
            except Exception as exc:  # noqa: BLE001 - surfaced to caller via job status
                log.exception("job %s failed", job_id)
                job["status"] = "error"
                job["error"] = str(exc)
                job["message"] = f"{job['stage']} で失敗"
                job["finished_at"] = now_iso()
                job["elapsed_s"] = round(asyncio.get_running_loop().time() - t0, 1)
                try:
                    await asyncio.to_thread(
                        self.repo.fail_run, job_id, job["error"], stage=job["stage"],
                        finished_at=job["finished_at"], elapsed_s=job["elapsed_s"],
                        study_instance_uid=job["study_instance_uid"],
                        series_instance_uid=job["series_instance_uid"])
                except Exception:  # noqa: BLE001
                    log.exception("could not record failure of job %s in DB", job_id)
            finally:
                self._persist(job)
                self.queue.task_done()

    # -- stages ---------------------------------------------------------------------

    def _set_stage(self, job: dict, stage: str, message: str) -> None:
        changed = job["stage"] != stage
        job["stage"], job["message"] = stage, message
        log.info("job %s [%s] %s", job["job_id"], stage, message)
        if changed:
            try:
                self.repo.update_run(job["job_id"], stage=stage)
            except Exception:  # noqa: BLE001 - progress bookkeeping must not kill the job
                log.warning("could not update stage of job %s in DB", job["job_id"])

    async def _run(self, job: dict) -> None:
        s = self.settings
        job_dir = s.jobs_dir / job["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)

        # 1) PACS -> NIfTI (blocking I/O + ITK -> worker thread)
        self._set_stage(job, "ingesting", "PACS から画像を取得中")
        ingest = await asyncio.to_thread(
            self.pacs.ingest, job["case_id"], job["study_instance_uid"],
            lambda m: self._set_stage(job, "ingesting", m))
        job["study_instance_uid"] = ingest.study_instance_uid
        job["series_instance_uid"] = ingest.series_instance_uid
        job["study_date"], job["modality"] = ingest.study_date, ingest.modality
        await asyncio.to_thread(
            self.repo.update_run, job["job_id"], study_instance_uid=ingest.study_instance_uid,
            series_instance_uid=ingest.series_instance_uid, study_date=ingest.study_date,
            modality=ingest.modality)
        (job_dir / "ingest.json").write_text(json.dumps({
            "study_instance_uid": ingest.study_instance_uid,
            "series_instance_uid": ingest.series_instance_uid,
            "cached": ingest.cached, "report": ingest.report,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        # 2) analysis
        ct_path = ingest.nifti_path
        if s.analysis_live and not ct_path.is_file():
            raise FileNotFoundError(f"CT (NIfTI) not found for case_id={job['case_id']}")

        config = engine.build_config(s, job_dir, job["case_id"], ct_path, job["levels"])
        (job_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

        self._set_stage(job, "analyzing", "大動脈セグメンテーション・計測を実行中"
                        + ("（実CT・約13分）" if s.analysis_live else "（MOCK）"))
        result = await engine.run_engine(s, job_dir, config, mock_delay_s=s.mock_delay_s)

        # 3) register (results + metadata in one transaction)
        self._set_stage(job, "registering", "解析結果を DB に登録中")
        finished = now_iso()
        elapsed = round((datetime.fromisoformat(finished)
                         - datetime.fromisoformat(job["started_at"])).total_seconds(), 1)
        await asyncio.to_thread(
            self.repo.complete_run, job["job_id"], result.rows,
            module_version=self.module_version,
            study_instance_uid=ingest.study_instance_uid,
            series_instance_uid=ingest.series_instance_uid,
            aorta_model=engine.model_name(s.aorta_model) if s.analysis_live else "mock",
            tl_model=engine.model_name(s.tl_model) if s.analysis_live else "mock",
            slicer_version=result.slicer_version,
            engine_backend=s.engine_backend if s.analysis_live else "mock",
            finished_at=finished, elapsed_s=elapsed)
        job["results"] = result.rows
        job["finished_at"], job["elapsed_s"] = finished, elapsed

    def _persist(self, job: dict) -> None:
        job_dir = self.settings.jobs_dir / job["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2),
                                          encoding="utf-8")


def _signed_url(settings: Settings, path, download_name: str) -> str:
    """V4 signed GET link for a file under data_dir (a GCS mount), signed through the IAM
    signBlob API as the attached runtime service account (no key file)."""
    from datetime import timedelta

    import google.auth
    from google.auth.transport.requests import Request
    from google.cloud import storage

    creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    obj = f"{settings.artifact_prefix}/{path.relative_to(settings.data_dir).as_posix()}"
    blob = storage.Client(project=project, credentials=creds).bucket(settings.artifact_bucket).blob(obj)
    return blob.generate_signed_url(
        version="v4", expiration=timedelta(minutes=15), method="GET",
        service_account_email=creds.service_account_email, access_token=creds.token,
        response_disposition=f'attachment; filename="{download_name}"')


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.analysis_live:
            settings.validate_live_analysis()  # fail fast: list everything that is missing
            print("=== LIVE MODE: Slicer WILL be executed (~13 min per job, GPU exclusive) ===",
                  flush=True)
            print("Starting in 3 seconds - Ctrl+C now to abort if this is unintended.", flush=True)
            await asyncio.sleep(3)
        else:
            print("=== MOCK MODE: Slicer will NOT be executed ===", flush=True)
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        runner = JobRunner(settings)
        n = runner.repo.mark_interrupted()
        if n:
            print(f"marked {n} unfinished run(s) from a previous process as error", flush=True)
        print(f"DB: {runner.repo.engine.url.render_as_string(hide_password=True)}", flush=True)
        try:
            print(f"PACS ({runner.pacs.mode}): {runner.pacs.ping()}", flush=True)
        except PacsError as exc:
            print(f"WARNING: PACS unreachable at startup: {exc}", flush=True)
        app.state.runner = runner
        task = asyncio.create_task(runner.worker())
        yield
        task.cancel()

    app = FastAPI(title="Med-Ops Analysis Job Server", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": "LIVE" if settings.analysis_live else "MOCK"}

    @app.post("/jobs", response_model=JobCreateResponse)
    async def create_job(req: JobCreateRequest):
        try:
            case_id = validate_case_id(req.case_id)
            levels = normalize_levels(req.levels)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        runner: JobRunner = app.state.runner
        try:
            job = await asyncio.to_thread(runner.submit, case_id, levels, req.study_instance_uid)
        except DailyLimitError as exc:
            raise HTTPException(429, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("submit failed")
            raise HTTPException(503, f"ジョブを登録できません（DB）: {exc}") from exc
        runner.queue.put_nowait(job["job_id"])
        return JobCreateResponse(job_id=job["job_id"], status="queued",
                                 queue_position=runner.pending_ahead(job["job_id"]))

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str):
        job = await asyncio.to_thread(app.state.runner.get, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return JobStatusResponse(**job)

    @app.get("/jobs", response_model=list[JobStatusResponse])
    async def list_jobs(limit: int = 20):
        runner: JobRunner = app.state.runner
        df = await asyncio.to_thread(runner.repo.runs, limit)
        out = []
        for job_id in df["job_id"]:
            job = await asyncio.to_thread(runner.get, job_id)
            if job:
                out.append(JobStatusResponse(**job))
        return out

    # ---- analysis view (images for the UI only; the LLM never receives these) ----

    def _job_dir(job_id: str):
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(400, "invalid job_id")
        return settings.jobs_dir / job_id

    @app.get("/jobs/{job_id}/views")
    async def job_views(job_id: str):
        job_dir = _job_dir(job_id)
        if not views.has_views(job_dir):
            raise HTTPException(404, "この解析には解析画面の画像がありません（機能追加前の解析など）")
        v = await asyncio.to_thread(views.render, job_dir)
        scene = job_dir / SCENE_FILE
        b64 = lambda b: base64.b64encode(b).decode() if b else None  # noqa: E731
        return {"job_id": job_id,
                "sections": [{"level": s["level"], "fov_mm": s["fov_mm"], "png_b64": b64(s["png"])}
                             for s in v["sections"]],
                "cpr_png_b64": b64(v["cpr"]), "cpr_levels": v["cpr_levels"],
                "scene_mb": round(scene.stat().st_size / 2**20, 1) if scene.is_file() else None}

    @app.get("/jobs/{job_id}/scene-link")
    async def job_scene_link(job_id: str):
        """Where the UI should send the browser for the scene file.

        Cloud Run caps a buffered response at 32 MiB and scenes are ~200 MB, so there the
        browser gets a 15-minute signed GCS link. Locally the UI streams /scene itself."""
        scene = _job_dir(job_id) / SCENE_FILE
        if not scene.is_file():
            raise HTTPException(404, "この解析には保存されたシーンがありません")
        if not settings.artifact_bucket:
            return {"url": None}
        job = await asyncio.to_thread(app.state.runner.get, job_id)
        name = f"{job['case_id']}_{job_id[:8]}_scene.mrb" if job else SCENE_FILE
        try:
            url = await asyncio.to_thread(_signed_url, settings, scene, name)
        except Exception as exc:  # noqa: BLE001 - e.g. missing signBlob permission
            log.exception("signing the scene link failed")
            raise HTTPException(503, f"ダウンロードリンクを発行できません: {type(exc).__name__}") from exc
        return {"url": url}

    @app.get("/jobs/{job_id}/scene")
    async def job_scene(job_id: str):
        scene = _job_dir(job_id) / SCENE_FILE
        if not scene.is_file():
            raise HTTPException(404, "この解析には保存されたシーンがありません")
        job = await asyncio.to_thread(app.state.runner.get, job_id)
        name = f"{job['case_id']}_{job_id[:8]}_scene.mrb" if job else SCENE_FILE
        return FileResponse(scene, media_type="application/octet-stream", filename=name)

    # ---- PACS (identifiers and series-level technical tags only) ----

    @app.get("/pacs/cases")
    async def pacs_cases():
        try:
            ids = await asyncio.to_thread(app.state.runner.pacs.list_case_ids)
        except PacsError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"mode": app.state.runner.pacs.mode, "case_ids": ids}

    @app.get("/pacs/studies")
    async def pacs_studies():
        """Catalog: case_id, study date, modalities, image counts (no names, no free text)."""
        try:
            studies = await asyncio.to_thread(app.state.runner.pacs.list_studies)
        except PacsError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"mode": app.state.runner.pacs.mode, "studies": studies}

    @app.get("/pacs/cases/{case_id}")
    async def pacs_case(case_id: str):
        try:
            validate_case_id(case_id)
            studies = await asyncio.to_thread(app.state.runner.pacs.find_studies, case_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except PacsError as exc:
            raise HTTPException(502, str(exc)) from exc
        if not studies:
            raise HTTPException(404, f"case_id={case_id} は PACS にありません")
        return {"case_id": case_id, "studies": [s.to_public() for s in studies]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Med-Ops analysis job server")
    parser.add_argument("--live", action="store_true", help="run real 3D Slicer (GPU, ~13 min/case)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings(live=True if args.live else None)
    if settings.analysis_live:
        settings.validate_live_analysis()  # fail before binding the port
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
