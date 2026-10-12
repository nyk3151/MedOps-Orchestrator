"""Registration and aggregation API over the schema.

Statistics use the LATEST successful run per case (re-analysis must not double-count N),
and are always split by analysis_mode so MOCK numbers never mix into real measurements.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import Integer, func, insert, select, update
from sqlalchemy.engine import Engine

from medops.db.schema import analysis_results, analysis_runs, init_db, make_engine
from medops.settings import Settings

RESULT_COLUMNS = ["aorta_area_mm2", "aorta_dia_mm", "tl_area_mm2", "fl_area_mm2", "tl_ratio_pct"]
STAT_METRICS = {
    "aorta_dia_mm": "大動脈径 (mm)",
    "aorta_area_mm2": "大動脈断面積 (mm²)",
    "tl_ratio_pct": "TL比 (%)",
}
CSV_RENAME = {"tl_ratio_pct": "TL%"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


class Repository:
    def __init__(self, engine: Engine):
        self.engine = engine

    @classmethod
    def from_settings(cls, settings: Settings, create: bool = True) -> "Repository":
        engine = make_engine(settings.database_url)
        if create:
            init_db(engine)
        return cls(engine)

    # ---------------------------------------------------------------- run lifecycle

    def create_run(self, *, job_id: str, case_id: str, levels: Iterable[str], analysis_mode: str,
                   module_name: str, created_at=None) -> int:
        with self.engine.begin() as c:
            res = c.execute(insert(analysis_runs).values(
                job_id=job_id, case_id=case_id, levels=",".join(levels), status="queued",
                stage="queued", analysis_mode=analysis_mode, module_name=module_name,
                created_at=_parse_ts(created_at) or _now()))
            return int(res.inserted_primary_key[0])

    def update_run(self, job_id: str, **fields) -> None:
        for k in ("started_at", "finished_at", "created_at"):
            if k in fields:
                fields[k] = _parse_ts(fields[k])
        with self.engine.begin() as c:
            c.execute(update(analysis_runs).where(analysis_runs.c.job_id == job_id).values(**fields))

    def complete_run(self, job_id: str, rows: list[dict], **run_fields) -> int:
        """Insert the per-level results and flip the run to success in ONE transaction."""
        for k in ("started_at", "finished_at"):
            if k in run_fields:
                run_fields[k] = _parse_ts(run_fields[k])
        with self.engine.begin() as c:
            run = c.execute(select(analysis_runs.c.run_id, analysis_runs.c.case_id)
                            .where(analysis_runs.c.job_id == job_id)).one()
            if rows:
                c.execute(insert(analysis_results), [
                    {"run_id": run.run_id, "case_id": run.case_id, "level": r["level"],
                     **{col: (None if r.get(col) is None else float(r[col])) for col in RESULT_COLUMNS}}
                    for r in rows])
            c.execute(update(analysis_runs).where(analysis_runs.c.job_id == job_id)
                      .values(status="success", stage="done", error_message=None, **run_fields))
            return int(run.run_id)

    def fail_run(self, job_id: str, error: str, **run_fields) -> None:
        self.update_run(job_id, status="error", error_message=error[-4000:], **run_fields)

    def mark_interrupted(self) -> int:
        """At server start: queued/running rows belong to a dead process."""
        with self.engine.begin() as c:
            res = c.execute(update(analysis_runs)
                            .where(analysis_runs.c.status.in_(("queued", "running")))
                            .values(status="error", finished_at=_now(),
                                    error_message="ジョブサーバーの再起動により中断されました"))
            return res.rowcount or 0

    # ---------------------------------------------------------------- reads

    def get_run(self, job_id: str) -> Optional[dict]:
        with self.engine.connect() as c:
            row = c.execute(select(analysis_runs).where(analysis_runs.c.job_id == job_id)).mappings().first()
            if row is None:
                return None
            run = dict(row)
            res = c.execute(select(analysis_results).where(analysis_results.c.run_id == run["run_id"])
                            .order_by(analysis_results.c.id)).mappings().all()
        run["results"] = [{k: r[k] for k in ["level", *RESULT_COLUMNS]} for r in res] or None
        return run

    def runs(self, limit: int = 50) -> pd.DataFrame:
        q = select(analysis_runs).order_by(analysis_runs.c.run_id.desc()).limit(limit)
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all(), columns=list(analysis_runs.c.keys()))

    def latest_results(self, analysis_mode: Optional[str] = None,
                       case_id: Optional[str] = None) -> pd.DataFrame:
        """One row per (case, level) from the latest successful run of each case."""
        r = analysis_runs
        latest = (select(r.c.case_id, r.c.analysis_mode, func.max(r.c.run_id).label("run_id"))
                  .where(r.c.status == "success")
                  .group_by(r.c.case_id, r.c.analysis_mode))
        if analysis_mode:
            latest = latest.where(r.c.analysis_mode == analysis_mode)
        if case_id:
            latest = latest.where(r.c.case_id == case_id)
        latest = latest.subquery()
        q = (select(analysis_results.c.case_id, analysis_results.c.level,
                    *[analysis_results.c[col] for col in RESULT_COLUMNS],
                    r.c.run_id, r.c.job_id, r.c.analysis_mode, r.c.study_date, r.c.modality,
                    r.c.series_instance_uid,
                    r.c.module_version, r.c.aorta_model, r.c.tl_model, r.c.slicer_version, r.c.engine_backend,
                    r.c.finished_at)
             .select_from(analysis_results
                          .join(latest, analysis_results.c.run_id == latest.c.run_id)
                          .join(r, r.c.run_id == analysis_results.c.run_id))
             .order_by(analysis_results.c.case_id, analysis_results.c.id))
        with self.engine.connect() as c:
            df = pd.DataFrame(c.execute(q).mappings().all(), columns=[col.name for col in q.selected_columns])
        return df

    def level_stats(self, analysis_mode: Optional[str] = None,
                    levels: Optional[list[str]] = None) -> pd.DataFrame:
        """N / mean / SD (sample) / median / min / max per level x metric."""
        df = self.latest_results(analysis_mode)
        cols = ["analysis_mode", "level", "metric", "label", "n", "mean", "sd", "median", "min", "max"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        if levels:
            df = df[df["level"].isin(levels)]
        out = []
        for (mode, level), g in df.groupby(["analysis_mode", "level"], sort=False):
            for metric, label in STAT_METRICS.items():
                s = pd.to_numeric(g[metric], errors="coerce").dropna()
                out.append({
                    "analysis_mode": mode, "level": level, "metric": metric, "label": label,
                    "n": int(s.size),
                    "mean": round(float(s.mean()), 2) if s.size else None,
                    "sd": round(float(s.std(ddof=1)), 2) if s.size > 1 else None,
                    "median": round(float(s.median()), 2) if s.size else None,
                    "min": round(float(s.min()), 2) if s.size else None,
                    "max": round(float(s.max()), 2) if s.size else None,
                })
        stats = pd.DataFrame(out, columns=cols)
        order = {lv: i for i, lv in enumerate(_level_order(stats["level"].unique()))}
        return stats.sort_values(["analysis_mode", "level", "metric"],
                                 key=lambda s: s.map(order) if s.name == "level" else s,
                                 ignore_index=True)

    def export_csv(self, analysis_mode: Optional[str] = None) -> bytes:
        """utf-8-sig CSV (opens correctly in Excel). tl_ratio_pct is written as 'TL%'."""
        df = self.latest_results(analysis_mode).rename(columns=CSV_RENAME)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8-sig")

    def successful_runs(self, case_id: str, analysis_mode: str) -> list[dict]:
        """Successful runs of one case, newest first (for picking an analysis to review)."""
        r = analysis_runs
        q = (select(r.c.run_id, r.c.job_id, r.c.finished_at, r.c.study_date, r.c.modality,
                    r.c.levels)
             .where(r.c.case_id == case_id, r.c.analysis_mode == analysis_mode,
                    r.c.status == "success")
             .order_by(r.c.run_id.desc()))
        with self.engine.connect() as c:
            rows = [dict(x) for x in c.execute(q).mappings().all()]
        for x in rows:
            if x["finished_at"] is not None and x["finished_at"].tzinfo is None:
                x["finished_at"] = x["finished_at"].replace(tzinfo=timezone.utc)  # SQLite
        return rows

    def count_runs_since(self, since: datetime, analysis_mode: str) -> int:
        r = analysis_runs
        with self.engine.connect() as c:
            return int(c.execute(select(func.count()).where(
                r.c.created_at >= since, r.c.analysis_mode == analysis_mode)).scalar() or 0)

    def signature(self) -> tuple:
        """Cheap change detector for the dashboard's auto-refresh (one small query)."""
        r = analysis_runs
        with self.engine.connect() as c:
            row = c.execute(select(func.max(r.c.run_id), func.count(),
                                   func.sum((r.c.status == "success").cast(Integer)),
                                   func.max(r.c.finished_at))).one()
        return tuple(str(x) for x in row)

    def counts(self) -> dict:
        with self.engine.connect() as c:
            rows = c.execute(select(analysis_runs.c.status, func.count())
                             .group_by(analysis_runs.c.status)).all()
        return {status: n for status, n in rows}


def _level_order(levels) -> list[str]:
    region = {"C": 0, "T": 1, "L": 2}
    return sorted(levels, key=lambda lv: (region.get(lv[0], 9), int(lv[1:]) if lv[1:].isdigit() else 99))
