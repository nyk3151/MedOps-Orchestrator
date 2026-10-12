"""Streamlit UI smoke test (headless AppTest) against a MOCK job server."""
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "medops" / "ui" / "app.py")


def _md(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_dashboard_and_chat_flow(server):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception
    assert "MOCK" in "".join(c.value for c in at.caption)
    assert any("まだ解析結果がありません" in i.value for i in at.info)

    at.chat_input[0].set_value("demo_003 を解析して").run()
    assert "実行してよろしいですか" in _md(at)
    at.chat_input[0].set_value("はい").run()
    assert "job_id は" in _md(at)

    time.sleep(1.5)
    at.run()
    assert not at.exception, at.exception
    assert at.metric[0].value == "1"           # N cases
    assert len(at.dataframe) >= 1              # stats table rendered
    assert any(b.label.startswith("⬇ CSV") for b in at.get("download_button"))
