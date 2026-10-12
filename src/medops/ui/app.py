"""Med-Ops Orchestrator UI: left = statistics dashboard, right = command box (chat).

    streamlit run src/medops/ui/app.py        (job server must be running; see README)

Auto-refresh design: the dashboard is a fragment re-run every few seconds. Each run issues
one tiny query (Repository.signature); the heavy reads are cached on that signature, so
the dashboard only really reloads when a run changes state. The chat lives outside the
fragment, so typing is never interrupted by the refresh. Job progress is read straight
from the job server (no LLM calls are spent on polling).
"""
from __future__ import annotations

import base64
import re
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

from medops.agent.chat import get_agent
from medops.auth import check_login, load_users
from medops.db.repo import STAT_METRICS, Repository
from medops.gcp_auth import IdTokenAuth
from medops.settings import SettingsError, load_settings

REFRESH_ACTIVE_S = 3     # while a job is queued/running: follow its progress
REFRESH_IDLE_S = 30      # otherwise: only pick up jobs started outside this UI
LIVE_ANALYSIS_EST_S = 13 * 60
STAGE_LABEL = {"queued": "キュー待ち", "ingesting": "PACS取得・変換", "analyzing": "解析",
               "registering": "DB登録", "done": "完了"}
STAGE_FRAC = {"queued": 0.02, "ingesting": 0.1, "analyzing": 0.3, "registering": 0.95}
SERIES_BLUE = "#2a78d6"     # categorical slot 1 (single series)
CSV_NAME_RE = re.compile(r"^medops_results_\d{8}_\d{6}\.csv$")

st.set_page_config(page_title="Med-Ops Orchestrator", layout="wide", page_icon="🩺")
# The top-right "running" animation would play on every background refresh; the chat has
# its own spinner, so hide Streamlit's global status widget.
st.markdown("<style>[data-testid='stStatusWidget']{visibility:hidden;}</style>",
            unsafe_allow_html=True)


# ------------------------------------------------------------------ shared resources

@st.cache_resource
def _settings():
    return load_settings()


@st.cache_resource
def _repo():
    return Repository.from_settings(_settings())


@st.cache_data(show_spinner=False)
def _load(signature: tuple, mode: str | None):
    repo = _repo()
    return repo.level_stats(mode), repo.latest_results(mode), repo.runs(30)


def _job_server(path: str):
    try:
        r = requests.get(f"{_settings().job_server_url}{path}", timeout=5, auth=IdTokenAuth())
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _ago(ts: str | None) -> float | None:
    if not ts:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()


# ------------------------------------------------------------------ header

try:
    settings = _settings()
    repo = _repo()
except SettingsError as exc:
    st.error(str(exc))
    st.stop()


# ------------------------------------------------------------------ login (only if accounts exist)

def _login_gate() -> None:
    users = load_users()
    if not users or st.session_state.get("user"):
        return
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.markdown("### 🩺 Med-Ops Orchestrator")
        st.caption("ログインしてください")
        with st.form("login"):
            username = st.text_input("ID")
            password = st.text_input("パスワード", type="password")
            submitted = st.form_submit_button("ログイン", width="stretch")
        if submitted:
            fails = st.session_state.get("login_fails", 0)
            time.sleep(min(2 ** fails, 30) if fails else 0)  # slow down guessing
            if check_login(username, password, users):
                st.session_state["user"] = username.strip()
                st.session_state["login_fails"] = 0
                st.rerun()
            st.session_state["login_fails"] = fails + 1
            st.error("ID またはパスワードが違います。")
    st.stop()


_login_gate()

health = _job_server("/health")
engine_mode = health["mode"] if health else "停止中"
DATA_MODE = "live" if (health["mode"] == "LIVE" if health else settings.analysis_live) else "mock"
st.markdown("### 🩺 Med-Ops Orchestrator")
st.caption(
    f"解析エンジン: **{engine_mode}** ・ PACS: **{settings.pacs_mode.upper()}** ・ "
    f"LLM: **{'Gemini ' + settings.gemini_model if settings.llm_mode == 'gemini' else 'MOCK（ルールベース）'}** ・ "
    f"DB: **{settings.database_url.split(':', 1)[0]}**"
)
if st.session_state.get("user"):
    if st.sidebar.button(f"ログアウト（{st.session_state['user']}）"):
        st.session_state.clear()
        st.rerun()
if not health:
    st.warning(f"解析ジョブサーバー（{settings.job_server_url}）に接続できません。"
               "`python -m medops.analysis.server` を起動してください。")

left, right = st.columns([3, 2], gap="large")


# ------------------------------------------------------------------ dashboard (left)

def _jobs_panel(jobs: list[dict]) -> None:
    active = [j for j in jobs if j["status"] in ("queued", "running")]
    if not active:
        return
    st.markdown("**実行中のジョブ**")
    for j in sorted(active, key=lambda j: j["created_at"]):
        stage = j.get("stage") or "queued"
        frac = STAGE_FRAC.get(stage, 0.0)
        elapsed = _ago(j.get("started_at"))
        if stage == "analyzing" and j.get("analysis_mode") == "live" and elapsed:
            frac = min(0.3 + 0.6 * elapsed / LIVE_ANALYSIS_EST_S, 0.9)
        when = f"・経過 {elapsed / 60:.1f} 分" if elapsed else ""
        st.progress(frac, text=f"**{j['case_id']}** — {STAGE_LABEL.get(stage, stage)}: "
                               f"{j.get('message') or ''}{when}")


def _notify_finished(jobs: list[dict]) -> None:
    seen = st.session_state.setdefault("seen_finished", None)
    finished = {j["job_id"]: j for j in jobs if j["status"] in ("success", "error")}
    if seen is not None:
        for jid in finished.keys() - seen:
            j = finished[jid]
            if j["status"] == "success":
                st.toast(f"解析完了: {j['case_id']}（ダッシュボードを更新しました）", icon="✅")
            else:
                st.toast(f"解析失敗: {j['case_id']} — {(j.get('error') or '')[:80]}", icon="⚠️")
    st.session_state["seen_finished"] = set(finished)


def _dot_plot(df: pd.DataFrame, metric: str, title: str) -> alt.Chart:
    d = df[["case_id", "level", metric]].dropna().copy()
    # Deterministic jitter from case_id: identical data must redraw identically (no flicker).
    d["jitter"] = d["case_id"].map(lambda c: (zlib.crc32(c.encode()) % 1000) / 1000 - 0.5)
    order = list(dict.fromkeys(d["level"]))
    # Explicit domain from the data: with N=1 the stdev layer is NaN and would skew it.
    lo, hi = float(d[metric].min()), float(d[metric].max())
    pad = max((hi - lo) * 0.15, abs(hi) * 0.05, 1.0)
    yscale = alt.Scale(domain=[lo - pad, hi + pad], nice=True, zero=False)
    base = alt.Chart(d).encode(x=alt.X("level:N", sort=order, title=None,
                                       axis=alt.Axis(labelAngle=0)))
    points = base.mark_circle(size=70, opacity=0.85, color=SERIES_BLUE, stroke="white",
                              strokeWidth=1).encode(
        y=alt.Y(f"{metric}:Q", title=title, scale=yscale),
        xOffset=alt.XOffset("jitter:Q", scale=alt.Scale(domain=[-1.5, 1.5])),
        tooltip=[alt.Tooltip("case_id:N", title="case_id"), alt.Tooltip("level:N", title="レベル"),
                 alt.Tooltip(f"{metric}:Q", title=title, format=".1f")],
    )
    err = base.mark_errorbar(extent="stdev", thickness=2, color="#52514e").encode(
        y=alt.Y(f"{metric}:Q", title=title, scale=yscale))
    mean = base.mark_tick(thickness=2, size=26, color="#52514e").encode(
        y=alt.Y(f"mean({metric}):Q", scale=yscale),
        tooltip=[alt.Tooltip("level:N", title="レベル"),
                 alt.Tooltip(f"mean({metric}):Q", title="平均", format=".1f"),
                 alt.Tooltip(f"stdev({metric}):Q", title="SD", format=".1f"),
                 alt.Tooltip(f"count({metric}):Q", title="N")])
    layers = points + mean
    if d.groupby("level")[metric].count().min() >= 2:  # SD needs N >= 2
        layers = points + err + mean
    return layers.properties(height=260, title=f"{title}（点=症例、線=平均±SD）")


def dashboard() -> None:
    jobs = _job_server("/jobs?limit=20") or []
    active = any(j["status"] in ("queued", "running") for j in jobs)
    if active != st.session_state.get("has_active", False):
        # Switch the refresh interval (fast while jobs run, slow when idle) with one full
        # rerun. Done before the toast so the toast is emitted by the run that completes.
        st.session_state["has_active"] = active
        st.rerun(scope="app")
    _notify_finished(jobs)
    _jobs_panel(jobs)

    # Only the data of the mode the job server runs in: real measurements on a LIVE
    # deployment, demo values in the zero-config quick start. Never both, no toggle.
    mode = DATA_MODE
    sig = repo.signature()
    stats, results, runs = _load(sig, mode)
    runs = runs[runs["analysis_mode"] == mode]

    n_cases = results["case_id"].nunique() if not results.empty else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("解析済み症例 (N)", n_cases)
    k2.metric("成功 run", int((runs["status"] == "success").sum()) if not runs.empty else 0)
    k3.metric("実行中/待ち", sum(1 for j in jobs if j["status"] in ("queued", "running")))
    k4.metric("失敗 run", int((runs["status"] == "error").sum()) if not runs.empty else 0)

    if results.empty:
        st.info("まだ解析結果がありません。右のチャットから解析を依頼してください（例: 「症例一覧を見せて」）。")
    else:
        st.markdown("**レベル別統計**（症例ごとの最新の成功 run を使用。SD は標本標準偏差）")
        table = stats.rename(columns={"level": "レベル", "label": "指標", "n": "N", "mean": "平均",
                                      "sd": "SD", "median": "中央値", "min": "最小", "max": "最大"})
        table = table[["レベル", "指標", "N", "平均", "SD", "中央値", "最小", "最大"]].astype(object)
        st.dataframe(table.where(table.notna(), "—"), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.altair_chart(_dot_plot(results, "aorta_dia_mm", STAT_METRICS["aorta_dia_mm"]),
                        width="stretch")
        if results["tl_ratio_pct"].notna().any():
            c2.altair_chart(_dot_plot(results, "tl_ratio_pct", STAT_METRICS["tl_ratio_pct"]),
                            width="stretch")
        else:
            c2.info("TL比のデータはありません（解離なし症例）。")
        st.download_button("⬇ CSV エクスポート（症例×レベル）", repo.export_csv(mode),
                           file_name="medops_results.csv", mime="text/csv",
                           width="stretch")
        with st.expander("症例別の計測値"):
            st.dataframe(results.rename(columns={"tl_ratio_pct": "TL%"})[
                ["case_id", "study_date", "modality", "level", "aorta_area_mm2", "aorta_dia_mm",
                 "tl_area_mm2", "fl_area_mm2", "TL%"]], hide_index=True, width="stretch")

    with st.expander("実行履歴・トレーサビリティ（analysis_runs）"):
        if runs.empty:
            st.caption("実行履歴はありません。")
        else:
            view = runs[["run_id", "case_id", "study_date", "modality", "status", "stage", "levels",
                         "series_instance_uid", "module_name", "module_version", "aorta_model",
                         "tl_model", "slicer_version", "engine_backend", "elapsed_s", "finished_at",
                         "error_message"]]
            st.dataframe(view, hide_index=True, width="stretch")


with left:
    interval = REFRESH_ACTIVE_S if st.session_state.get("has_active") else REFRESH_IDLE_S
    st.fragment(run_every=interval)(dashboard)()


# ------------------------------------------------------------------ chat (right)

def _agent():
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = get_agent(settings)
        except SettingsError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state.messages = []
    return st.session_state.agent


@st.cache_data(show_spinner=False, max_entries=20)
def _views(job_id: str) -> dict | None:
    """Rendered analysis screen of one job (immutable once the job finished)."""
    return _job_server(f"/jobs/{job_id}/views")


@st.cache_data(show_spinner=False, ttl=600, max_entries=20)
def _scene_link_cached(job_id: str) -> dict:
    """Signed links live 15 min; cache 10 min so the page doesn't re-sign on every rerun."""
    link = _job_server(f"/jobs/{job_id}/scene-link")
    if link is None:
        raise LookupError(job_id)  # failures are not cached: retry on the next rerun
    return link


def _scene_link(job_id: str) -> dict | None:
    try:
        return _scene_link_cached(job_id)
    except LookupError:
        return None


def _scene_bytes(job_id: str):
    def fetch() -> bytes:
        r = requests.get(f"{_settings().job_server_url}/jobs/{job_id}/scene", timeout=600,
                         auth=IdTokenAuth())
        r.raise_for_status()
        return r.content
    return fetch


def _fmt(v, unit: str) -> str:
    return "—" if v is None else f"{v:g} {unit}"


def _analysis_view(res: dict) -> None:
    job_id = res["analysis_view"]
    v = _views(job_id)
    st.markdown(f"**解析画面** — 症例 {res['case_id']}・解析 {res.get('analyzed_at') or '—'}"
                f"（検査日 {res.get('study_date') or '—'}）")
    if not v:
        st.caption("この解析には画像が保存されていません（解析画面の機能を追加する前の解析です）。")
        return
    rows = {r["level"]: r for r in res.get("results") or []}
    cols = st.columns(max(len(v["sections"]), 1))
    for col, sec in zip(cols, v["sections"]):
        r = rows.get(sec["level"], {})
        col.image(base64.b64decode(sec["png_b64"]), width="stretch")
        col.caption(f"**{sec['level']}** 計測断面（{sec['fov_mm']:g} mm 四方）  \n"
                    f"大動脈 {_fmt(r.get('aorta_area_mm2'), 'mm²')} ・ 真腔 {_fmt(r.get('tl_area_mm2'), 'mm²')}"
                    f" ・ TL比 {_fmt(r.get('tl_ratio_pct'), '%')}")
    if v.get("cpr_png_b64"):
        st.image(base64.b64decode(v["cpr_png_b64"]), width="stretch")
        marks = "、".join(m["level"] for m in v.get("cpr_levels", []))
        st.caption(f"CPR 像（大動脈の中心線に沿って伸ばした長軸像。左が大動脈起始部側、"
                   f"黄色の縦線が計測レベル: {marks or '—'}）")
    st.caption("色: 赤 = 真腔、青 = 偽腔、黄 = 大動脈壁の輪郭")
    if v.get("scene_mb"):
        local = settings.jobs_dir / job_id / "scene.mrb"
        if settings.slicer_exe and Path(settings.slicer_exe).is_file() and local.is_file():
            if st.button("🧠 3D Slicer で開く", key=f"slicer-{job_id}"):
                subprocess.Popen([str(settings.slicer_exe), str(local)])  # noqa: S603 - fixed args
                st.toast("3D Slicer を起動しました（シーンの読み込みに少しかかります）。")
        label = f"⬇ Slicer シーン（.mrb, {v['scene_mb']:g} MB）"
        link = _scene_link(job_id)
        if link and link.get("url"):   # Cloud Run: straight from storage (no 32 MiB cap)
            st.link_button(label, link["url"], key=f"mrb-{job_id}")
        else:                          # local: stream through the job server
            st.download_button(label, _scene_bytes(job_id),
                               file_name=f"{res['case_id']}_{job_id[:8]}_scene.mrb",
                               mime="application/octet-stream", key=f"mrb-{job_id}")


def _render_tool_calls(calls: list[dict]) -> None:
    if not calls:
        return
    with st.expander(f"🔧 ツール呼び出し {len(calls)} 件"):
        for c in calls:
            ok = not (isinstance(c["result"], dict) and c["result"].get("error"))
            st.markdown(f"{'✅' if ok else '⚠️'} `{c['tool']}` ({c['ms']} ms)")
            st.json({"args": c["args"], "result": c["result"]}, expanded=False)
    for c in calls:
        res = c["result"] if isinstance(c["result"], dict) else {}
        if c["tool"] == "show_analysis_view" and res.get("analysis_view"):
            _analysis_view(res)
    for i, c in enumerate(calls):
        res = c["result"] if isinstance(c["result"], dict) else {}
        for j, link in enumerate(res.get("links", [])):
            if str(link.get("url", "")).startswith(("http://", "https://")):
                st.link_button(f"↗ {link['label']}", link["url"], key=f"ln-{c['at']}-{i}-{j}")
    for c in calls:
        name = c["result"].get("file_name") if c["tool"] == "export_csv" and isinstance(c["result"], dict) else None
        if name and CSV_NAME_RE.match(name) and (settings.results_dir / name).is_file():
            st.download_button(f"⬇ {name}", (settings.results_dir / name).read_bytes(),
                               file_name=name, mime="text/csv", key=f"dl-{name}")


with right:
    agent = _agent()
    st.markdown(f"**指令box** <span style='opacity:.6'>— {agent.name}</span>", unsafe_allow_html=True)
    box = st.container(height=560)
    with box:
        if not st.session_state.messages:
            st.caption("例: 「症例一覧を見せて」「未解析の症例を T8 と T12 で解析して」"
                       "「T8 の統計を教えて」「CSV で出力して」")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                _render_tool_calls(m.get("tool_calls", []))

    if prompt := st.chat_input("自然言語で指示（例: 症例一覧を見せて）"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with box:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("考え中…（PACS / 解析エンジン / DB を操作しています）"):
                    try:
                        reply = agent.send(prompt)
                        text, calls = reply.text, reply.tool_calls
                    except Exception as exc:  # noqa: BLE001 - show, don't crash the UI
                        text, calls = f"⚠️ エラー: {exc}", []
        st.session_state.messages.append({"role": "assistant", "content": text, "tool_calls": calls})
        st.rerun()
