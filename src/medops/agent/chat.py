"""Chat agents: Gemini (Vertex AI function calling) and an offline rule-based MOCK agent.

Both expose ``send(text) -> AgentReply`` and drive the SAME tools (medops.agent.tools), so
the UI, the tool layer and everything below it behave identically with or without a Vertex
key. The MOCK agent exists so a fresh clone can demo the whole flow (CLAUDE.md: every layer
has a mock, MOCK by default).

Gemini: client.chats.create() + send_message (models.generate_content with function calling
emits deprecation warnings). Auth: service-account key via GOOGLE_APPLICATION_CREDENTIALS.
No ADC / gcloud auth.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Optional

from medops.agent import tools
from medops.agent.tools import TOOLS, drain_calls
from medops.settings import Settings, load_settings

SYSTEM_INSTRUCTION = """
あなたは医療画像解析パイプライン「Med-Ops Orchestrator」の司令塔です。
医師の自然言語の指示を、ツール呼び出し（PACS 検索 → 解析ジョブ → DB 集計）に変換します。

## 原則
- 扱えるのは匿名化された case_id・モダリティ・検査日（StudyDate）・枚数と、数値の解析結果だけです。
  画像・氏名などの個人情報は扱いません。
- 計算やファイルパスの扱いはしません。数値はツールの返り値をそのまま使い、推測で作りません。
- 症例の確認には list_pacs_cases / describe_case を使います。case_id は PACS 上の ID をそのまま使います。
- 「CT は何件？」「10月の CT は？」のような件数・期間の質問は search_studies を使います。
  「10月」「先月」などは今日の日付をもとに YYYY-MM-DD の範囲に直して渡し、
  回答では対象期間・検査数・症例数を示してください。
- 「PACS を開いて」「002 の画像を見せて」は open_pacs_viewer を使います。リンクはチャット欄に
  ボタンとして表示されるので、URL は書き写さず「下のボタンから開けます」と伝えてください。
- 「002、2026/10/12 の解析画面を出して」は show_analysis_view（date は YYYY-MM-DD）。
  画像はチャット欄に自動で表示されます。あなたは画像を見られないので、画像の内容を説明せず、
  返り値の数値（レベルごとの面積・TL比）を簡潔に添えてください。該当がなければ available の日付を案内します。

## 解析の実行（重い処理）
- run_analysis は GPU を占有します（LIVE の実CTで 1 症例あたり約 13 分、1 件ずつ順番に実行）。
  所要時間は list_pacs_cases の analysis_engine（MOCK なら数秒）から見積もってください。
- 呼び出す前に必ず「対象 case_id・件数・おおよその所要時間」を伝えて確認し、
  ユーザーが明確に同意した場合のみ user_confirmed=True で呼び出してください。
- 「未解析の症例を全部」のような依頼は list_pacs_cases の not_analyzed を使います。
- 1 Study に CT シリーズが複数ある症例は解析できません（推測で選ばない方針）。
  describe_case の analyzable=false のときは理由（シリーズ一覧）を説明してください。
- 開始したら job_id を明記し、「進捗は画面に表示され、完了するとダッシュボードが更新される」と伝えます。
  自分からポーリングはしません。ユーザーに聞かれたら get_job_status / list_recent_jobs を使います。

## 集計・出力
- 統計は get_statistics。レベル（T8 / T12）ごとに N・平均・標準偏差を示してください。
- CSV は export_csv。ファイルはチャット欄のボタンからダウンロードできると伝えます。

回答は簡潔な日本語で。数値は単位付きで、表が適切なら Markdown の表を使ってください。
""".strip()


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------------------- Gemini

def build_chat(settings: Optional[Settings] = None):
    from google import genai
    from google.genai import types

    settings = settings or load_settings()
    settings.validate_gemini()
    client = genai.Client(
        vertexai=True,
        project=settings.gcp_project,
        location=settings.gcp_location,
    )
    chat = client.chats.create(
        model=settings.gemini_model,
        config=types.GenerateContentConfig(
            tools=TOOLS,
            system_instruction=f"{SYSTEM_INSTRUCTION}\n\n今日の日付: {tools.today_jst()}（日本時間）",
            temperature=0.2,
        ),
    )
    # Chat only holds the shared api_client transitively (via Models), not a
    # reference to `client` itself. Without this, `client` gets garbage
    # collected as soon as build_chat() returns, and genai.Client.__del__
    # closes the underlying httpx client out from under this chat, breaking
    # every send_message() after the first with "client has been closed".
    chat._keepalive_client = client
    return chat


class GeminiAgent:
    name = "Gemini"

    def __init__(self, settings: Settings):
        self.model = settings.gemini_model
        self.chat = build_chat(settings)

    def send(self, text: str) -> AgentReply:
        drain_calls()
        response = self.chat.send_message(text)
        return AgentReply(response.text or "（応答が空でした）", drain_calls())


# ------------------------------------------------------------------------------- MOCK

_TOKEN = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]+(?![A-Za-z0-9_-])")
_LEVEL = re.compile(r"(?<![A-Za-z0-9])([CTLctl](?:1[0-2]|[1-9]))(?![0-9])")
_YES = re.compile(r"^(はい|お願い|おねがい|ok|yes|y|実行|了解|どうぞ|いいよ|いいです|進めて|よろしく)", re.I)
_NO = re.compile(r"^(いいえ|no|n|やめ|キャンセル|中止|待って)", re.I)


def _fmt(v, unit=""):
    return "—" if v is None else f"{v:g}{unit}"


class MockAgent:
    """Deterministic keyword agent. Same tools, same confirmation rule, no LLM."""

    name = "MOCK"
    model = "rule-based"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pending: Optional[dict] = None
        self.last_job_ids: list[str] = []

    # -- helpers

    def _known_cases(self) -> list[str]:
        res = tools._call("GET", "/pacs/cases")
        return res.get("case_ids", []) if isinstance(res, dict) else []

    def _case_ids(self, text: str) -> list[str]:
        known = set(self._known_cases())
        out = []
        for tok in _TOKEN.findall(text):
            if (tok in known or re.fullmatch(r"(case|demo|phantom)[_-]?\w+", tok, re.I)) \
                    and not _LEVEL.fullmatch(tok) and tok not in out:
                out.append(tok)
        return out

    @staticmethod
    def _levels(text: str) -> Optional[list[str]]:
        found = [m.upper() for m in _LEVEL.findall(text)]
        return list(dict.fromkeys(found)) or None

    # -- turn

    def send(self, text: str) -> AgentReply:
        drain_calls()
        reply = self._respond(text.strip())
        return AgentReply(reply, drain_calls())

    def _respond(self, text: str) -> str:
        if self.pending:
            if _YES.search(text):
                p, self.pending = self.pending, None
                return self._start(p["case_ids"], p["levels"])
            if _NO.search(text):
                self.pending = None
                return "解析は実行しませんでした。"

        levels = self._levels(text)
        cases = self._case_ids(text)
        wants_run = re.search(r"解析|計測|測定|analy", text, re.I) and not re.search(r"結果|統計", text)

        if re.search(r"csv|エクスポート|出力|ダウンロード", text, re.I):
            r = tools.export_csv()
            return r.get("error") or f"CSV を作成しました（{r['rows']} 行）。下のボタンからダウンロードできます。"
        if re.search(r"統計|平均|標準偏差|集計|まとめ|分布", text):
            return self._stats(levels)
        if re.search(r"解析画面|解析結果の画面|断面", text) and cases:
            d = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
            day = f"{d.group(1)}-{int(d.group(2)):02d}-{int(d.group(3)):02d}" if d else None
            r = tools.show_analysis_view(case_id=cases[0], date=day)
            if "error" in r:
                return r["error"]
            return (f"{r['case_id']} の解析画面です（解析 {r['analyzed_at']}）。"
                    "下に各レベルの計測断面と CPR 像を表示します。")
        if re.search(r"pacs|ビューア|ビューワ|画像を(見|表示|開)", text, re.I) \
                and re.search(r"開|起動|見|表示", text):
            r = tools.open_pacs_viewer(case_id=cases[0] if cases else None)
            return r.get("error") or "下のボタンから PACS を開けます。" + r["note"]
        if re.search(r"どのくらい|何件|件数|いくつ|(?<![A-Za-z])(CT|MR|MRI)(?![A-Za-z])", text, re.I) \
                and not wants_run:
            return self._search(text)
        if wants_run and re.search(r"未解析|全部|すべて|全て|残り", text):
            r = tools.list_pacs_cases()
            if "error" in r:
                return r["error"]
            if not r["not_analyzed"]:
                return "未解析の症例はありません。"
            return self._propose(r["not_analyzed"], levels)
        if wants_run and cases:
            return self._propose(cases, levels)
        if re.search(r"結果", text) and cases:
            return self._results(cases[0])
        if re.search(r"状況|進捗|状態|終わ|どうな|ジョブ", text):
            return self._status()
        if re.search(r"詳細|シリーズ|study|series", text, re.I) and cases:
            return self._describe(cases[0])
        if re.search(r"一覧|症例|リスト|pacs|どんな|ありますか", text, re.I):
            return self._list()
        if cases:
            return self._describe(cases[0])
        return ("MOCK エージェント（キーワード応答）です。例:\n"
                "- 「症例一覧を見せて」\n- 「demo_001 を解析して」/「未解析の症例を全部 T8 と T12 で解析して」\n"
                "- 「進捗は？」\n- 「T8 の統計を出して」\n- 「CSV で出力して」\n\n"
                "自然な会話で操作するには `llm_mode: gemini` を設定してください。")

    def _list(self) -> str:
        r = tools.list_pacs_cases()
        if "error" in r:
            return r["error"]
        lines = [f"PACS（{r['pacs_mode']}）の症例は {r['count']} 件です。", "",
                 "| case_id | 検査日 | モダリティ | 解析 |", "|---|---|---|---|"]
        lines += [f"| {c['case_id']} | {', '.join(c['study_dates']) or '—'} | "
                  f"{'/'.join(c['modalities']) or '—'} | {'済' if c['analyzed'] else '未'} |"
                  for c in r["cases"]]
        return "\n".join(lines)

    def _search(self, text: str) -> str:
        m = re.search(r"(?<![A-Za-z])(CT|MR|MRI|PT|NM|CR|DX|US)(?![A-Za-z])", text, re.I)
        modality = {"MRI": "MR"}.get(m.group(1).upper(), m.group(1).upper()) if m else None
        lo = hi = None
        if mo := re.search(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月", text):
            year = int(mo.group(1) or tools.today_jst()[:4])
            month = int(mo.group(2))
            if 1 <= month <= 12:
                last = (date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)).day
                lo, hi = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"
        r = tools.search_studies(modality=modality, date_from=lo, date_to=hi)
        if "error" in r:
            return r["error"]
        what = (f"{lo[:7]} の " if lo else "") + (f"{modality} " if modality else "全モダリティの")
        lines = [f"{what}検査は {r['study_count']} 件（{r['case_count']} 症例、{r['image_count']} 枚）です。"]
        if r["studies"]:
            lines += ["", "| case_id | 検査日 | モダリティ | 枚数 | 解析 |", "|---|---|---|---|---|"]
            lines += [f"| {s['case_id']} | {s['study_date'] or '—'} | {'/'.join(s['modalities'])} | "
                      f"{s['image_count']} | {'済' if s['analyzed'] else '未'} |" for s in r["studies"]]
        return "\n".join(lines)

    def _describe(self, case_id: str) -> str:
        r = tools.describe_case(case_id)
        if "error" in r:
            return r["error"]
        lines = [f"**{case_id}**: Study {len(r['studies'])} 件"]
        for st in r["studies"]:
            for s in st["series"]:
                mark = "候補" if s["is_candidate"] else "候補外"
                lines.append(f"- {s['modality']} {s['num_instances']}枚 / 厚さ {s['slice_thickness']} / "
                             f"{s['image_type']}（{mark}）")
        lines.append("→ 解析可能です。" if r["analyzable"] else
                     "→ **解析できません**（1 Study = 1 CT シリーズのみ対応。推測でシリーズを選びません）。")
        return "\n".join(lines)

    def _propose(self, case_ids: list[str], levels: Optional[list[str]]) -> str:
        ok, ng = [], []
        for cid in case_ids:
            d = tools.describe_case(cid)
            (ok if d.get("analyzable") else ng).append((cid, d.get("error")))
        msg = []
        if ng:
            msg.append("次の症例は解析できません: " + ", ".join(
                f"{c}（{e or '候補シリーズが1件ではありません'}）" for c, e in ng))
        if not ok:
            return "\n".join(msg) or "対象の症例がありません。"
        health = tools._call("GET", "/health")
        live = isinstance(health, dict) and health.get("mode") == "LIVE"
        ids = [c for c, _ in ok]
        lv = levels or ["T8", "T12"]
        est = f"約 {13 * len(ids)} 分（GPU で 1 件ずつ）" if live else "数秒（MOCK 解析）"
        self.pending = {"case_ids": ids, "levels": lv}
        msg.append(f"{', '.join(ids)}（{len(ids)} 件）を {' / '.join(lv)} で解析します。"
                   f"所要時間は{est}です。実行してよろしいですか？（はい / いいえ）")
        return "\n".join(msg)

    def _start(self, case_ids: list[str], levels: list[str]) -> str:
        r = tools.run_analysis(case_ids=case_ids, levels=levels, user_confirmed=True)
        if "error" in r:
            return r["error"]
        self.last_job_ids = [j["job_id"] for j in r["jobs"] if "job_id" in j]
        lines = [f"解析を開始しました（{r['analysis_mode']}）。"]
        lines += [f"- {j['case_id']}: " + (f"job_id は {j['job_id']} です" if "job_id" in j
                                           else f"エラー: {j['error']}") for j in r["jobs"]]
        lines.append("進捗は画面に表示され、完了するとダッシュボードが自動更新されます。")
        return "\n".join(lines)

    def _status(self) -> str:
        r = tools.list_recent_jobs(limit=5)
        if "error" in r:
            return r["error"]
        if not r["jobs"]:
            return "ジョブはまだありません。"
        lines = ["| case_id | 状態 | 段階 | 経過(s) |", "|---|---|---|---|"]
        lines += [f"| {j['case_id']} | {j['status']} | {j['stage']} | {_fmt(j['elapsed_s'])} |"
                  for j in r["jobs"]]
        return "直近のジョブです。\n\n" + "\n".join(lines)

    def _results(self, case_id: str) -> str:
        r = tools.get_case_results(case_id)
        if "error" in r:
            return r["error"]
        if not r["results"]:
            return r["message"]
        out = [f"**{case_id}**", "", "| レベル | 径 | 面積 | TL比 |", "|---|---|---|---|"]
        out += [f"| {x['level']} | {_fmt(x['aorta_dia_mm'], ' mm')} | {_fmt(x['aorta_area_mm2'], ' mm²')} "
                f"| {_fmt(x['tl_ratio_pct'], ' %')} |" for x in r["results"]]
        return "\n".join(out)

    def _stats(self, levels: Optional[list[str]]) -> str:
        r = tools.get_statistics(levels=levels)
        if "error" in r:
            return r["error"]
        if not r["stats"]:
            return r["message"]
        lines = ["| レベル | 指標 | N | 平均 | SD |", "|---|---|---|---|---|"]
        names = {"aorta_dia_mm": "径 (mm)", "aorta_area_mm2": "面積 (mm²)", "tl_ratio_pct": "TL比 (%)"}
        lines += [f"| {s['level']} | {names[s['metric']]} | {s['n']} | "
                  f"{_fmt(s['mean'])} | {_fmt(s['sd'])} |" for s in r["stats"]]
        return "\n".join(lines)


def get_agent(settings: Optional[Settings] = None):
    settings = settings or load_settings()
    tools.configure(settings)
    return GeminiAgent(settings) if settings.llm_mode == "gemini" else MockAgent(settings)
