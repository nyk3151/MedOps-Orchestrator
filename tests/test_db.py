"""DB layer. Runs on SQLite; set MEDOPS_TEST_DATABASE_URL to also run against PostgreSQL:

    MEDOPS_TEST_DATABASE_URL=postgresql+psycopg://medops:medops@127.0.0.1:5432/medops_test pytest tests/test_db.py

(The database name must end in _test: the fixture drops all tables.)
"""
import io
import os
import uuid

import pandas as pd
import pytest
from sqlalchemy import text

from medops.db.repo import Repository
from medops.db.schema import init_db, make_engine, metadata


def _urls():
    urls = ["sqlite"]
    if os.environ.get("MEDOPS_TEST_DATABASE_URL_ORIG"):
        urls.append(os.environ["MEDOPS_TEST_DATABASE_URL_ORIG"])
    return urls


# conftest strips MEDOPS_* vars, so capture the PG url at import time.
if os.environ.get("MEDOPS_TEST_DATABASE_URL"):
    os.environ["MEDOPS_TEST_DATABASE_URL_ORIG"] = os.environ["MEDOPS_TEST_DATABASE_URL"]


@pytest.fixture(params=_urls())
def repo(request, tmp_path):
    url = f"sqlite:///{(tmp_path / 't.db').as_posix()}" if request.param == "sqlite" else request.param
    engine = make_engine(url)
    if request.param != "sqlite" and not (engine.url.database or "").endswith("_test"):
        # the fixture drops all tables: never point it at a real database
        pytest.exit(f"MEDOPS_TEST_DATABASE_URL must name a *_test database "
                    f"(got {engine.url.database!r})", returncode=2)
    if request.param != "sqlite":
        metadata.drop_all(engine)
    init_db(engine)
    yield Repository(engine)
    if request.param != "sqlite":
        metadata.drop_all(engine)
    engine.dispose()


def _row(level, area, tl=None):
    r = {"level": level, "aorta_area_mm2": area, "aorta_dia_mm": round(2 * (area / 3.14159) ** 0.5, 2)}
    if tl is not None:
        r.update(tl_area_mm2=tl, fl_area_mm2=area - tl, tl_ratio_pct=round(tl / area * 100, 1))
    return r


def _run(repo, case_id, rows, mode="live", status="success"):
    job_id = uuid.uuid4().hex
    repo.create_run(job_id=job_id, case_id=case_id, levels=[r["level"] for r in rows] or ["T8"],
                    analysis_mode=mode, module_name="AorticPipeline")
    if status == "success":
        repo.complete_run(job_id, rows, module_version="v1.1", series_instance_uid="1.2.3",
                          aorta_model="Aorta_net_2", tl_model="Dissection_net",
                          slicer_version="5.8.1", elapsed_s=780.0)
    else:
        repo.fail_run(job_id, "boom")
    return job_id


def test_lifecycle_and_metadata(repo):
    job = _run(repo, "c1", [_row("T8", 500, 300), _row("T12", 400)])
    run = repo.get_run(job)
    assert run["status"] == "success" and run["stage"] == "done"
    assert run["series_instance_uid"] == "1.2.3" and run["module_version"] == "v1.1"
    assert run["aorta_model"] == "Aorta_net_2"  # identifier, not a path
    t12 = [r for r in run["results"] if r["level"] == "T12"][0]
    assert t12["tl_ratio_pct"] is None and t12["tl_area_mm2"] is None  # non-dissection: NULL
    assert isinstance(t12["aorta_area_mm2"], float)


def test_stats_use_latest_run_per_case_and_split_modes(repo):
    _run(repo, "c1", [_row("T8", 100), _row("T12", 100)])
    _run(repo, "c1", [_row("T8", 300), _row("T12", 300)])     # re-analysis supersedes
    _run(repo, "c2", [_row("T8", 500), _row("T12", 500)])
    _run(repo, "c3", [_row("T8", 9999)], status="error")      # failures never count
    _run(repo, "m1", [_row("T8", 700)], mode="mock")          # mock kept apart

    live = repo.level_stats("live")
    t8 = live[(live.level == "T8") & (live.metric == "aorta_area_mm2")].iloc[0]
    assert t8.n == 2 and t8["mean"] == 400.0 and t8["sd"] == pytest.approx(141.42, abs=0.01)
    assert list(live.level.unique()) == ["T8", "T12"]
    assert set(repo.level_stats().analysis_mode) == {"live", "mock"}
    assert len(repo.latest_results("live")) == 4


def test_csv_export(repo):
    _run(repo, "c1", [_row("T8", 500, 300)])
    df = pd.read_csv(io.BytesIO(repo.export_csv("live")), encoding="utf-8-sig")
    assert "TL%" in df.columns and "tl_ratio_pct" not in df.columns
    assert df.loc[0, "TL%"] == 60.0
    assert repo.export_csv("live").startswith("﻿".encode("utf-8"))


def test_unique_run_level_and_fk(repo):
    job = _run(repo, "c1", [_row("T8", 500)])
    with pytest.raises(Exception):
        repo.complete_run(job, [_row("T8", 1)])  # duplicate (run_id, level)
    with repo.engine.connect() as c:
        assert c.execute(text("select count(*) from analysis_results")).scalar() == 1


def test_mark_interrupted(repo):
    repo.create_run(job_id="j1", case_id="c1", levels=["T8"], analysis_mode="mock",
                    module_name="AorticPipeline")
    assert repo.mark_interrupted() == 1
    assert repo.get_run("j1")["status"] == "error"


def test_old_database_gets_new_columns(tmp_path):
    """A DB created before engine_backend existed is migrated in place (additive only)."""
    from sqlalchemy import create_engine, inspect

    url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
    old = create_engine(url)
    with old.begin() as c:
        c.execute(text("CREATE TABLE analysis_runs (run_id INTEGER PRIMARY KEY, job_id TEXT, "
                       "case_id TEXT, levels TEXT, status TEXT, analysis_mode TEXT, created_at TEXT)"))
    old.dispose()
    engine = make_engine(url)
    init_db(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("analysis_runs")}
    assert "engine_backend" in cols and "slicer_version" in cols
    engine.dispose()
