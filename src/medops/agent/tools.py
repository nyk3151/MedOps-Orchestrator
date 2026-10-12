"""Tools exposed to the orchestrator LLM (Gemini) and to the offline MOCK agent.

Rules (CLAUDE.md):
- Arguments are meaningful identifiers only (case_id, levels, job_id). Never paths.
- Return values are anonymized IDs, numbers, modality and study date only. No DICOM,
  images, names or other patient info (study date allowed: team decision 2026-10-12).
- Errors are returned as {"error": ...} so the model can explain them instead of crashing.

PACS and jobs go through the job-server API (it owns path resolution); results and
statistics are read from the DB, which the dashboard reads as well.
"""
# NOTE: no `from __future__ import annotations` here. google-genai converts tool arguments
# with isinstance() on the real annotation objects; string annotations break every tool call.
import functools
import json
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

import requests

from medops.db.repo import Repository
from medops.gcp_auth import IdTokenAuth
from medops.ids import normalize_levels, validate_case_id
from medops.settings import Settings, load_settings

log = logging.getLogger("medops.agent")

JST = timezone(timedelta(hours=9), "JST")  # dates the users speak in

_settings: Optional[Settings] = None
_repo: Optional[Repository] = None

# Every tool call is recorded per thread (Streamlit runs each session in its own thread,
# and the Gemini SDK executes tools synchronously in the calling thread).
_local = threading.local()


def _calls() -> list[dict]:
    if not hasattr(_local, "calls"):
        _local.calls = []
    return _local.calls


def drain_calls() -> list[dict]:
    """Return and clear the tool calls recorded in this thread."""
    calls = list(_calls())
    _calls().clear()
    return calls


def configure(settings: Settings, repo: Optional[Repository] = None) -> None:
    global _settings, _repo
    _settings, _repo = settings, repo


def _s() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def _db() -> Repository:
    global _repo
    if _repo is None:
        _repo = Repository.from_settings(_s())
    return _repo


def recorded(fn: Callable) -> Callable:
    """Record name/args/result/duration of every call; keep signature + docstring for Gemini."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tools never raise into the model
            log.exception("tool %s failed", fn.__name__)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        _calls().append({
            "tool": fn.__name__, "args": kwargs or dict(zip(fn.__code__.co_varnames, args)),
            "result": result, "ms": round((time.perf_counter() - t0) * 1000),
            "at": datetime.now(timezone.utc).isoformat(),
        })
        log.info("tool %s(%s) -> %s", fn.__name__, json.dumps(kwargs, ensure_ascii=False),
                 json.dumps(result, ensure_ascii=False, default=str)[:300])
        return result
    return wrapper


def _call(method: str, path: str, json_body: Optional[dict] = None) -> dict | list:
    base = _s().job_server_url
    try:
        resp = requests.request(method, f"{base}{path}", json=json_body, timeout=30,
                                auth=IdTokenAuth())
    except requests.RequestException:
        return {"error": f"解析ジョブサーバー ({base}) に接続できません。起動しているか確認してください。"}
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return {"error": f"HTTP {resp.status_code}: {detail}"}
    return resp.json()


def _engine_info() -> dict:
    health = _call("GET", "/health")
    live = isinstance(health, dict) and health.get("mode") == "LIVE"
    return {"analysis_mode": "LIVE" if live else "MOCK", "minutes_per_case": 13 if live else 0.1}


def _data_mode() -> str:
    """Results shown to the user are those of the mode the engine runs in (never mixed)."""
    return "live" if _engine_info()["analysis_mode"] == "LIVE" else "mock"


def _analyzed_case_ids() -> set[str]:
    df = _db().latest_results(_data_mode())
    return set() if df.empty else set(df["case_id"])


# ------------------------------------------------------------------------------- tools

def today_jst() -> str:
    return datetime.now(JST).date().isoformat()


def _iso_day(value: Optional[str], name: str) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value).strip().replace("/", "-")).isoformat()
    except ValueError:
        raise ValueError(f"{name} は YYYY-MM-DD 形式で指定してください: {value!r}") from None


@recorded
def list_pacs_cases() -> dict:
    """List every case_id stored in the PACS with its study date(s) and modalities, and whether
    each one already has analysis results.

    Use this when the user asks which cases exist, or which cases are not analyzed yet.
    """
    res = _call("GET", "/pacs/studies")
    if "error" in res:
        return res
    analyzed = _analyzed_case_ids()
    cases: dict[str, dict] = {}
    for st in res["studies"]:
        c = cases.setdefault(st["case_id"], {"case_id": st["case_id"], "study_dates": [],
                                             "modalities": [], "analyzed": st["case_id"] in analyzed})
        if st["study_date"] and st["study_date"] not in c["study_dates"]:
            c["study_dates"].append(st["study_date"])
        c["modalities"] = sorted(set(c["modalities"]) | set(st["modalities"]))
    rows = sorted(cases.values(), key=lambda c: c["case_id"])
    return {"pacs_mode": res["mode"], "count": len(rows), "cases": rows,
            "not_analyzed": [c["case_id"] for c in rows if not c["analyzed"]],
            "analysis_engine": _engine_info()}


@recorded
def search_studies(modality: Optional[str] = None, date_from: Optional[str] = None,
                   date_to: Optional[str] = None) -> dict:
    """Count and list the studies in the PACS by modality and/or study date (StudyDate).

    Use this for questions like "how many CT studies are there?" or "how many CT studies
    from October?". Convert relative expressions to explicit dates yourself (the result
    includes today's date); e.g. "10月" -> date_from="2026-10-01", date_to="2026-10-31".

    Args:
        modality: DICOM modality such as "CT" or "MR". Default: any modality.
        date_from: First study date to include, "YYYY-MM-DD". Default: no lower bound.
        date_to: Last study date to include, "YYYY-MM-DD". Default: no upper bound.
    """
    mod = (modality or "").strip().upper() or None
    if mod and not mod.isalnum():
        return {"error": f"modality が不正です: {modality!r}"}
    lo, hi = _iso_day(date_from, "date_from"), _iso_day(date_to, "date_to")
    res = _call("GET", "/pacs/studies")
    if "error" in res:
        return res
    analyzed = _analyzed_case_ids()
    hits = [st for st in res["studies"]
            if (not mod or mod in st["modalities"])
            and (not lo or (st["study_date"] or "") >= lo)
            and (not hi or (st["study_date"] or "9999") <= hi)]
    if lo or hi:  # a study without a date cannot be placed in a date range
        hits = [st for st in hits if st["study_date"]]
    by_modality: dict[str, int] = {}
    for st in hits:
        for m in st["modalities"]:
            by_modality[m] = by_modality.get(m, 0) + 1
    return {
        "today": today_jst(),
        "filter": {"modality": mod, "date_from": lo, "date_to": hi},
        "study_count": len(hits),
        "case_count": len({st["case_id"] for st in hits}),
        "image_count": sum(st["image_count"] for st in hits),
        "studies_by_modality": by_modality,
        "studies": [{"case_id": st["case_id"], "study_date": st["study_date"],
                     "modalities": st["modalities"], "image_count": st["image_count"],
                     "analyzed": st["case_id"] in analyzed} for st in hits[:200]],
    }


@recorded
def describe_case(case_id: str) -> dict:
    """Show the studies / series stored in the PACS for one case and whether it can be analyzed.

    Only 1 Study = 1 CT series cases are analyzable; scouts / dose reports are ignored.

    Args:
        case_id: Anonymized case identifier as stored in the PACS (DICOM PatientID).
    """
    validate_case_id(case_id)
    res = _call("GET", f"/pacs/cases/{case_id}")
    if "error" in res:
        return res
    studies = res["studies"]
    for st in studies:
        n = sum(1 for s in st["series"] if s["is_candidate"])
        st["analyzable"] = n == 1
        st["candidate_series"] = n
        for s in st["series"]:
            s.pop("series_description", None)  # free text: keep it out of the model
    return {"case_id": case_id, "studies": studies,
            "analyzable": len(studies) == 1 and studies[0]["analyzable"]}


@recorded
def open_pacs_viewer(case_id: Optional[str] = None) -> dict:
    """Open the PACS (Orthanc) in the browser: the study list, or one case's images.

    Use this when the user asks to open / launch / show the PACS or a case's images.
    The links are shown to the user as buttons in the chat.

    Args:
        case_id: Anonymized case identifier to open. Omit to open the whole study list.
    """
    s = _s()
    if s.pacs_mode != "live":
        return {"error": "MOCK PACS（デモ用の仮想 PACS）には画像ビューアがありません。"}
    base = s.pacs_viewer_url
    if not case_id:
        return {"links": [{"label": "PACS（検査一覧）", "url": f"{base}/ui/app/"}],
                "note": "ログインを求められたら PACS 用の ID / パスワードを入力してください。"}
    validate_case_id(case_id)
    res = _call("GET", f"/pacs/cases/{case_id}")
    if "error" in res:
        return res
    links = [{"label": f"PACS で {case_id} を開く",
              "url": f"{base}/ui/app/#/filtered-studies?PatientID={case_id}"}]
    for st in res["studies"]:
        day = st.get("study_date") or ""
        day = f"{day[:4]}-{day[4:6]}-{day[6:]}" if len(day) == 8 else "日付なし"
        links.append({"label": f"画像ビューア（{case_id}・{day}）",
                      "url": f"{base}/stone-webviewer/index.html?study={st['study_instance_uid']}"})
    return {"case_id": case_id, "links": links,
            "note": "ログインを求められたら PACS 用の ID / パスワードを入力してください（閲覧専用）。"}


@recorded
def show_analysis_view(case_id: str, date: Optional[str] = None) -> dict:
    """Show the analysis screen of a past analysis: the measured cross-section at each level
    (aorta / true lumen / false lumen overlaid) and the straightened long-axis (CPR) view.
    The images appear in the chat below your answer; you only receive identifiers and numbers.

    Use this for requests like "症例002、2026/10/12の解析画面をだして".

    Args:
        case_id: Anonymized case identifier.
        date: "YYYY-MM-DD". Matches the analysis date (Japan time) or the study date.
              Omit to show the latest analysis of the case.
    """
    validate_case_id(case_id)
    day = _iso_day(date, "date")
    runs = _db().successful_runs(case_id, _data_mode())
    if not runs:
        return {"error": f"症例 {case_id} の解析結果はまだありません。"}

    def analyzed_on(r):
        return r["finished_at"].astimezone(JST).date().isoformat() if r["finished_at"] else None

    hits = [r for r in runs if not day or day in (analyzed_on(r), r["study_date"])]
    if not hits:
        return {"error": f"{case_id} に {day} の解析はありません。",
                "available": [{"analyzed_on": analyzed_on(r), "study_date": r["study_date"]}
                              for r in runs]}
    run = hits[0]
    detail = _db().get_run(run["job_id"]) or {}
    return {"case_id": case_id, "job_id": run["job_id"], "run_id": run["run_id"],
            "analyzed_at": run["finished_at"].astimezone(JST).strftime("%Y-%m-%d %H:%M")
            if run["finished_at"] else None,
            "study_date": run["study_date"], "modality": run["modality"],
            "results": detail.get("results") or [],
            "analysis_view": run["job_id"],
            "note": "解析画面（各レベルの計測断面と CPR 像）はチャット欄に表示されます。"}


@recorded
def run_analysis(case_ids: list[str], levels: Optional[list[str]] = None,
                 user_confirmed: bool = False) -> dict:
    """Queue aortic CT analysis jobs (PACS fetch -> segmentation/measurement -> DB registration).

    Jobs run one at a time on the GPU (about 13 minutes per case on real CT). You must first
    tell the user how many cases and roughly how long it will take, and only call this with
    user_confirmed=True after they explicitly agree.

    Args:
        case_ids: One or more anonymized case identifiers, e.g. ["002"].
        levels: Vertebral levels to measure, e.g. ["T8", "T12"]. Defaults to ["T8", "T12"].
        user_confirmed: Must be True; set only after the user explicitly agreed to run.
    """
    if not user_confirmed:
        return {"error": "ユーザーの明示的な同意が必要です。件数と所要時間を伝えて確認してください。"}
    if isinstance(case_ids, str):
        case_ids = [case_ids]
    levels = normalize_levels(levels)
    engine = _engine_info()
    per_case_min = engine["minutes_per_case"]
    jobs = []
    for cid in case_ids:
        validate_case_id(cid)
        r = _call("POST", "/jobs", {"case_id": cid, "levels": levels})
        jobs.append({"case_id": cid, **({"error": r["error"]} if "error" in r
                                        else {"job_id": r["job_id"], "queue_position": r["queue_position"]})})
    queued = [j for j in jobs if "job_id" in j]
    return {"analysis_mode": engine["analysis_mode"], "levels": levels, "jobs": jobs,
            "estimated_total_minutes": round(per_case_min * (len(queued) + (queued[0]["queue_position"] if queued else 0)), 1),
            "note": "進捗は画面に自動表示されます。完了するとダッシュボードが自動更新されます。"}


@recorded
def get_job_status(job_id: str) -> dict:
    """Get the status, current stage and (once finished) the measurements of an analysis job.

    Args:
        job_id: The job id returned by run_analysis.
    """
    res = _call("GET", f"/jobs/{job_id}")
    if isinstance(res, dict) and "error" not in res:
        res = {k: res.get(k) for k in ("job_id", "case_id", "status", "stage", "message",
                                       "levels", "elapsed_s", "results", "error", "analysis_mode")}
    return res


@recorded
def list_recent_jobs(limit: int = 10) -> dict:
    """List the most recent analysis jobs (queued / running / success / error).

    Args:
        limit: How many jobs to return (default 10).
    """
    res = _call("GET", f"/jobs?limit={int(limit)}")
    if isinstance(res, dict):
        return res
    return {"jobs": [{k: j.get(k) for k in ("job_id", "case_id", "status", "stage",
                                             "elapsed_s", "analysis_mode", "error")} for j in res]}


@recorded
def get_case_results(case_id: str) -> dict:
    """Get the latest successful measurements of one case from the database.

    Args:
        case_id: Anonymized case identifier.
    """
    validate_case_id(case_id)
    df = _db().latest_results(_data_mode(), case_id=case_id)
    if df.empty:
        return {"case_id": case_id, "results": [], "message": "この症例の解析結果はまだありません。"}
    cols = ["level", "aorta_area_mm2", "aorta_dia_mm", "tl_area_mm2", "fl_area_mm2", "tl_ratio_pct"]
    return {"case_id": case_id, "job_id": df["job_id"].iloc[0],
            "module_version": df["module_version"].iloc[0],
            "results": json.loads(df[cols].to_json(orient="records"))}


@recorded
def get_statistics(levels: Optional[list[str]] = None) -> dict:
    """Cohort statistics per vertebral level: N, mean, SD, median, min, max of aortic diameter,
    aortic area and true-lumen ratio. Uses the latest successful analysis of each case.

    Args:
        levels: Restrict to these levels, e.g. ["T8"]. Default: all levels.
    """
    lv = normalize_levels(levels) if levels else None
    stats = _db().level_stats(_data_mode(), lv)
    if stats.empty:
        return {"message": "集計対象の解析結果がまだありません。", "stats": []}
    return {"stats": json.loads(stats.drop(columns=["label", "analysis_mode"]).to_json(orient="records")),
            "note": "sd は標本標準偏差。"}


@recorded
def export_csv() -> dict:
    """Export the latest results of all cases as a CSV file (UTF-8 with BOM, opens in Excel).
    The user downloads it from the chat / dashboard.
    """
    data = _db().export_csv(_data_mode())
    name = f"medops_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out = _s().results_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    rows = max(data.decode("utf-8-sig").count("\n") - 1, 0)
    return {"file_name": name, "rows": rows}


TOOLS = [list_pacs_cases, search_studies, describe_case, open_pacs_viewer, show_analysis_view,
         run_analysis, get_job_status, list_recent_jobs, get_case_results, get_statistics,
         export_csv]
