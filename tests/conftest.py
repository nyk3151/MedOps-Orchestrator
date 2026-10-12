import socket
import threading
import time

import pytest
import uvicorn

from medops import settings as settings_mod

_ENV_PREFIXES = ("MEDOPS_",)
_ENV_EXTRA = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Every test starts with zero config: no MEDOPS_* / GCP env, no settings.yaml, tmp data dir."""
    import os

    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES) or key in _ENV_EXTRA:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MEDOPS_SETTINGS", str(tmp_path / "no-settings.yaml"))
    monkeypatch.setenv("MEDOPS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEDOPS_MOCK_DELAY_S", "0")
    # The UI caches settings/repo with st.cache_resource, which outlives an AppTest run.
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()
    yield


@pytest.fixture
def settings():
    return settings_mod.load_settings()


@pytest.fixture
def server(monkeypatch):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    monkeypatch.setenv("MEDOPS_JOB_SERVER_URL", f"http://127.0.0.1:{port}")
    from medops.analysis.server import create_app

    settings = settings_mod.load_settings()
    srv = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port,
                                        log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    while not srv.started:
        time.sleep(0.05)
    yield settings
    srv.should_exit = True
    th.join(5)
