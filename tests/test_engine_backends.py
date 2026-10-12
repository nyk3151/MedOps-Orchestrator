"""Engine backends: config paths must be the ones the WORKER sees."""
import types
from pathlib import Path

import pytest

from medops.analysis import engine
from medops.settings import load_settings


def _settings(monkeypatch, tmp_path, backend, **env):
    a, t = tmp_path / "Aorta_net_2", tmp_path / "Dissection_net"
    a.mkdir(), t.mkdir()
    monkeypatch.setenv("MEDOPS_ENGINE_BACKEND", backend)
    monkeypatch.setenv("MEDOPS_AORTA_MODEL", str(a))
    monkeypatch.setenv("MEDOPS_TL_MODEL", str(t))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return load_settings(live=True)


def test_docker_config_uses_container_paths(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path, "docker", MEDOPS_WORKER_IMAGE="medops-gpu-worker:test")
    s.validate_live_analysis()  # no slicer_exe / module_dir needed for docker
    job_dir = s.jobs_dir / "abc123"
    cfg = engine.build_config(s, job_dir, "001", s.nifti_dir / "001_0000.nii.gz", ["T8"])
    assert cfg["input"] == "/data/nifti/001_0000.nii.gz"
    assert cfg["output_dir"] == "/data/jobs/abc123"
    assert cfg["module_dir"] == engine.WORKER_MODULE_DIR
    assert cfg["aorta_model"] == "/models/aorta" and cfg["tl_model"] == "/models/tl"

    cmd = engine._docker_cmd(s, job_dir, cfg)
    assert cmd[:4] == ["docker", "run", "--rm", "--gpus"] and "--shm-size" in cmd
    assert any(c.endswith("target=/models/aorta,readonly") for c in cmd)
    assert cmd[-2:] == ["medops-gpu-worker:test", "/data/jobs/abc123/config.json"]


def test_docker_refuses_paths_outside_data_dir(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path, "docker", MEDOPS_WORKER_IMAGE="x")
    with pytest.raises(engine.AnalysisError):
        engine.build_config(s, s.jobs_dir / "j", "001", tmp_path / "elsewhere.nii.gz", ["T8"])


def test_backend_specific_validation(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path, "cloudrun")
    with pytest.raises(Exception, match="gpu_job"):
        s.validate_live_analysis()


def test_cloudrun_executes_job_with_config_override(monkeypatch, tmp_path):
    s = _settings(monkeypatch, tmp_path, "cloudrun", MEDOPS_GPU_JOB="medops-gpu",
                  GOOGLE_CLOUD_PROJECT="proj")
    cfg = engine.build_config(s, Path("/mnt/medops/data/jobs/j1"), "001",
                              Path("/mnt/medops/data/nifti/001_0000.nii.gz"), ["T8"])
    assert cfg["input"] == "/mnt/medops/data/nifti/001_0000.nii.gz"

    calls = []

    class FakeSession:
        def __init__(self, creds):
            pass

        def post(self, url, json, timeout):
            calls.append((url, json))
            return types.SimpleNamespace(status_code=200, json=lambda: {"name": "operations/op1"})

        def get(self, url, timeout):
            return types.SimpleNamespace(json=lambda: {
                "name": "operations/op1", "done": True,
                "response": {"name": "projects/proj/locations/r/jobs/medops-gpu/executions/e1",
                             "succeededCount": 1}})

    monkeypatch.setattr(engine, "CLOUDRUN_POLL_S", 0)
    monkeypatch.setattr(engine, "_authorized_session", lambda: FakeSession(None))

    assert engine._run_cloudrun_job(s, cfg) == "execution=e1"
    url, body = calls[0]
    assert url.endswith("/projects/proj/locations/asia-southeast1/jobs/medops-gpu:run")
    assert body["overrides"]["containerOverrides"][0]["args"] == ["/mnt/medops/data/jobs/j1/config.json"]


def test_cloudrun_results_json_decides_not_exit_code(monkeypatch, tmp_path):
    """A job that crashes in Slicer's shutdown after writing results.json still succeeds;
    a failed job without results.json is an error that carries the job's message."""
    import asyncio
    import json

    s = _settings(monkeypatch, tmp_path, "cloudrun", MEDOPS_GPU_JOB="medops-gpu",
                  GOOGLE_CLOUD_PROJECT="proj")
    job_dir = s.jobs_dir / "j1"
    job_dir.mkdir(parents=True)
    cfg = engine.build_config(s, job_dir, "001", s.nifti_dir / "001_0000.nii.gz", ["T8"])
    failed = "Cloud Run Job が失敗しました: Task failed with exit code: 1"
    monkeypatch.setattr(engine, "_run_cloudrun_job", lambda *_: failed)

    async def no_wait(_):
        pass

    monkeypatch.setattr(engine.asyncio, "sleep", no_wait)
    with pytest.raises(engine.AnalysisError, match="exit code: 1"):
        asyncio.run(engine.run_engine(s, job_dir, cfg))

    (job_dir / "results.json").write_text(json.dumps({
        "status": "success", "case_id": "001", "slicer_version": "5.8.1",
        "results": [{"level": "Th8", "aorta_area_mm2": 650.0, "tl_area_mm2": 251.75,
                     "tl_ratio_%": 38.7}]}),
        encoding="utf-8")
    res = asyncio.run(engine.run_engine(s, job_dir, cfg))
    assert res.rows[0]["level"] == "T8" and res.slicer_version == "5.8.1"
