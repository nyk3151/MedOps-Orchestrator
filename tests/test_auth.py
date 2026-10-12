import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from medops.analysis.server import DailyLimitError, JobRunner
from medops.auth import check_login, hash_password, load_users, verify_password
from medops.settings import load_settings

APP = str(Path(__file__).resolve().parents[1] / "src" / "medops" / "ui" / "app.py")


def test_hash_and_verify():
    h = hash_password("s3cret")
    assert h.startswith("scrypt$") and "s3cret" not in h
    assert verify_password("s3cret", h)
    assert not verify_password("wrong", h)
    assert hash_password("s3cret") != h  # salted


def test_check_login_and_load_users():
    users = load_users(json.dumps({"judge": hash_password("pw")}))
    assert check_login("judge", "pw", users)
    assert check_login(" judge ", "pw", users)
    assert not check_login("judge", "nope", users)
    assert not check_login("nobody", "pw", users)
    assert load_users("") == {}


def test_ui_requires_login_only_when_accounts_exist(monkeypatch, server):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.text_input  # no accounts configured: no login (local quick start)

    monkeypatch.setenv("MEDOPS_UI_USERS", json.dumps({"judge": hash_password("pw")}))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert [t.label for t in at.text_input] == ["ID", "パスワード"]
    assert not at.chat_input  # nothing behind the gate is rendered

    at.text_input[0].set_value("judge")
    at.text_input[1].set_value("wrong")
    at.button[0].click().run()
    assert any("違います" in e.value for e in at.error)

    at.text_input[1].set_value("pw")
    at.button[0].click().run()
    assert at.chat_input and not at.exception


def test_daily_live_job_cap(monkeypatch):
    monkeypatch.setenv("MEDOPS_MAX_LIVE_JOBS_PER_DAY", "2")
    runner = JobRunner(load_settings(live=True))
    runner.submit("demo_001", ["T8"])
    runner.submit("demo_002", ["T8"])
    with pytest.raises(DailyLimitError):
        runner.submit("demo_003", ["T8"])
