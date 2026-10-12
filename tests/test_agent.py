"""Agent layer against a real (MOCK-mode) job server running in a background thread."""
import time

import pytest

from medops.agent import tools
from medops.agent.chat import MockAgent, get_agent
from medops.settings import load_settings


def _wait_idle(settings, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = tools.list_recent_jobs.__wrapped__(limit=50)["jobs"]
        if jobs and all(j["status"] in ("success", "error") for j in jobs):
            return jobs
        time.sleep(0.1)
    raise AssertionError("jobs did not finish")


def test_tools_never_leak_paths_or_names(server):
    tools.configure(server)
    out = repr([tools.list_pacs_cases(), tools.describe_case("demo_001"),
                tools.run_analysis(case_ids=["demo_001"], user_confirmed=True)])
    _wait_idle(server)
    out += repr([tools.list_recent_jobs(), tools.get_case_results("demo_001"),
                 tools.get_statistics(), tools.export_csv()])
    for needle in (str(server.data_dir), server.data_dir.as_posix(), "nii.gz", ".dcm",
                   "PatientName", "series_description"):
        assert needle not in out, needle


def test_run_requires_confirmation(server):
    tools.configure(server)
    assert "error" in tools.run_analysis(case_ids=["demo_001"])


def test_mock_agent_full_flow(server):
    agent = get_agent(server)
    assert isinstance(agent, MockAgent)

    r = agent.send("症例一覧を見せて")
    assert "demo_001" in r.text and r.tool_calls[0]["tool"] == "list_pacs_cases"

    r = agent.send("demo_001 と demo_multi を T8 で解析して")
    assert "demo_multi" in r.text and "解析できません" in r.text
    assert "実行してよろしいですか" in r.text
    assert not any(c["tool"] == "run_analysis" for c in r.tool_calls)  # not before consent

    r = agent.send("はい")
    assert "job_id は" in r.text
    call = [c for c in r.tool_calls if c["tool"] == "run_analysis"][0]
    assert call["args"]["user_confirmed"] is True and call["args"]["case_ids"] == ["demo_001"]
    assert call["args"]["levels"] == ["T8"]
    _wait_idle(server)

    r = agent.send("demo_001 の結果は？")
    assert "T8" in r.text and "mm" in r.text

    r = agent.send("未解析の症例を全部解析して")
    assert "demo_002" in r.text and "demo_001" not in r.text.split("を")[0]
    r = agent.send("いいえ")
    assert "実行しませんでした" in r.text

    r = agent.send("統計を出して")
    assert "| T8 |" in r.text and "mock" not in r.text

    r = agent.send("CSVで出力して")
    assert "CSV を作成しました" in r.text
    assert list(server.results_dir.glob("medops_results_*.csv"))


def test_agent_reports_job_server_down(monkeypatch):
    monkeypatch.setenv("MEDOPS_JOB_SERVER_URL", "http://127.0.0.1:9")
    agent = get_agent(load_settings())
    assert "接続できません" in agent.send("症例一覧").text


def test_tool_signatures_are_usable_by_gemini_sdk():
    """google-genai's AFC converts args from real annotations (regression: string annotations)."""
    from google.genai import _extra_utils

    samples = {"describe_case": {"case_id": "c1"},
               "run_analysis": {"case_ids": ["c1"], "levels": ["T8"], "user_confirmed": True},
               "get_job_status": {"job_id": "x"}, "list_recent_jobs": {"limit": 3},
               "get_case_results": {"case_id": "c1"},
               "get_statistics": {"levels": ["T8"]},
               "export_csv": {}, "list_pacs_cases": {},
               "search_studies": {"modality": "CT", "date_from": "2026-10-01", "date_to": "2026-10-31"},
               "open_pacs_viewer": {"case_id": "c1"},
               "show_analysis_view": {"case_id": "c1", "date": "2026-10-12"}}
    for fn in tools.TOOLS:
        _extra_utils.convert_argument_from_function(samples[fn.__name__], fn)


def test_search_studies_by_modality_and_date(server):
    tools.configure(load_settings())
    r = tools.search_studies(modality="ct")
    assert r["filter"]["modality"] == "CT" and r["study_count"] == 9 and r["case_count"] == 9
    assert r["image_count"] > 0 and r["studies_by_modality"] == {"CT": 9}
    assert all(set(s) == {"case_id", "study_date", "modalities", "image_count", "analyzed"}
               for s in r["studies"])  # identifiers, dates and counts only

    october = tools.search_studies(modality="CT", date_from="2026-10-01", date_to="2026/10/31")
    assert {s["case_id"] for s in october["studies"]} == {"demo_005", "demo_006", "demo_007",
                                                         "demo_008", "demo_multi"}
    assert all("2026-10-01" <= s["study_date"] <= "2026-10-31" for s in october["studies"])
    assert tools.search_studies(modality="MR")["study_count"] == 0
    assert "error" in tools.search_studies(date_from="10月")

    cases = tools.list_pacs_cases()["cases"]
    assert cases[0]["study_dates"] == ["2026-09-03"] and cases[0]["modalities"] == ["CT"]


def test_mock_agent_answers_count_questions(server):
    agent = get_agent(load_settings())
    r = agent.send("CT画像はどのくらいありますか？")
    assert "CT 検査は 9 件" in r.text and r.tool_calls[0]["tool"] == "search_studies"
    r = agent.send("2026年10月のCTデータはどのくらいある？")
    assert "2026-10 の CT 検査は 5 件" in r.text


def test_open_pacs_viewer_links(monkeypatch, server):
    tools.configure(load_settings())
    assert "error" in tools.open_pacs_viewer("demo_001")  # MOCK PACS: nothing to open

    monkeypatch.setenv("MEDOPS_PACS_MODE", "live")
    monkeypatch.setenv("MEDOPS_PACS_VIEWER_URL", "https://pacs.example/")
    tools.configure(load_settings())
    monkeypatch.setattr(tools, "_call", lambda method, path, body=None: {
        "case_id": "002", "studies": [{"study_instance_uid": "1.2.3", "study_date": "20261012"}]})
    r = tools.open_pacs_viewer("002")
    urls = [l["url"] for l in r["links"]]
    assert urls == ["https://pacs.example/ui/app/#/filtered-studies?PatientID=002",
                    "https://pacs.example/stone-webviewer/index.html?study=1.2.3"]
    assert "2026-10-12" in r["links"][1]["label"]
    assert tools.open_pacs_viewer()["links"][0]["url"] == "https://pacs.example/ui/app/"
    assert "error" in tools.open_pacs_viewer("../x")


def test_analysis_view_after_mock_run(server):
    import base64
    import requests
    from datetime import datetime, timedelta, timezone

    agent = get_agent(load_settings())
    agent.send("demo_003 を解析して")
    agent.send("はい")
    _wait_idle(server)

    today = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    r = tools.show_analysis_view("demo_003", today.replace("-", "/"))
    assert r["analysis_view"] and [x["level"] for x in r["results"]] == ["T8", "T12"]
    assert "png" not in str(r).lower()  # the model only gets ids and numbers
    miss = tools.show_analysis_view("demo_003", "2001-01-01")
    assert "error" in miss and miss["available"][0]["analyzed_on"] == today

    v = requests.get(f"{server.job_server_url}/jobs/{r['job_id']}/views", timeout=30).json()
    assert [s["level"] for s in v["sections"]] == ["T8", "T12"]
    assert base64.b64decode(v["sections"][0]["png_b64"])[1:4] == b"PNG"
    assert v["cpr_png_b64"] and [m["level"] for m in v["cpr_levels"]] == ["T8", "T12"]
    assert requests.get(f"{server.job_server_url}/jobs/..%2Fx/views", timeout=30).status_code in (400, 404)

    reply = agent.send(f"demo_003、{today.replace('-', '/')}の解析画面をだして")
    assert "解析画面" in reply.text and reply.tool_calls[0]["tool"] == "show_analysis_view"
