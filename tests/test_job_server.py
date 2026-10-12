import time

from fastapi.testclient import TestClient

from medops.analysis.server import create_app


def _wait(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("success", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_mock_job_end_to_end(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["mode"] == "MOCK"
        r = client.post("/jobs", json={"case_id": "demo_001"})
        assert r.status_code == 200
        job = _wait(client, r.json()["job_id"])
        assert job["status"] == "success", job
        assert [row["level"] for row in job["results"]] == ["T8", "T12"]
        row = job["results"][0]
        assert set(row) >= {"aorta_area_mm2", "aorta_dia_mm", "tl_ratio_pct"}
        job_dir = settings.jobs_dir / job["job_id"]
        assert (job_dir / "config.json").is_file()   # kept for reproducibility
        assert (job_dir / "results.json").is_file()
        assert (job_dir / "ingest.json").is_file()
        assert job["series_instance_uid"]


def test_multi_series_case_fails_without_guessing(settings):
    with TestClient(create_app(settings)) as client:
        job = _wait(client, client.post("/jobs", json={"case_id": "demo_multi"}).json()["job_id"])
        assert job["status"] == "error"
        assert job["stage"] == "ingesting"
        assert "候補シリーズが 2 件" in job["error"]
        assert job["results"] is None


def test_unknown_case_fails(settings):
    with TestClient(create_app(settings)) as client:
        job = _wait(client, client.post("/jobs", json={"case_id": "nope"}).json()["job_id"])
        assert job["status"] == "error" and "PACS にありません" in job["error"]


def test_pacs_endpoints(settings):
    with TestClient(create_app(settings)) as client:
        assert "demo_001" in client.get("/pacs/cases").json()["case_ids"]
        case = client.get("/pacs/cases/demo_001").json()
        series = case["studies"][0]["series"]
        assert [s["is_candidate"] for s in series] == [False, True]
        assert client.get("/pacs/cases/nope").status_code == 404
        assert client.get("/pacs/cases/a%2Fb").status_code in (400, 404)


def test_rejects_bad_ids(settings):
    with TestClient(create_app(settings)) as client:
        assert client.post("/jobs", json={"case_id": "../etc"}).status_code == 400
        assert client.post("/jobs", json={"case_id": "c1", "levels": ["X1"]}).status_code == 400


def test_success_is_registered_in_db_and_survives_restart(settings):
    from medops.db.repo import Repository

    with TestClient(create_app(settings)) as client:
        job = _wait(client, client.post("/jobs", json={"case_id": "demo_002"}).json()["job_id"])
        assert job["status"] == "success" and job["message"].startswith("完了")
    with TestClient(create_app(settings)) as client:  # new process state
        again = client.get(f"/jobs/{job['job_id']}").json()
        assert again["status"] == "success" and again["results"] == job["results"]
        assert again["series_instance_uid"] == job["series_instance_uid"]
        assert client.get("/jobs").json()[0]["job_id"] == job["job_id"]
    run = Repository.from_settings(settings).get_run(job["job_id"])
    assert run["analysis_mode"] == "mock" and run["module_name"] == "AorticPipeline"
    assert run["elapsed_s"] is not None and run["finished_at"] is not None


def test_failure_is_registered_in_db(settings):
    from medops.db.repo import Repository

    with TestClient(create_app(settings)) as client:
        job = _wait(client, client.post("/jobs", json={"case_id": "demo_multi"}).json()["job_id"])
    run = Repository.from_settings(settings).get_run(job["job_id"])
    assert run["status"] == "error" and run["stage"] == "ingesting"
    assert "候補シリーズ" in run["error_message"] and run["results"] is None
