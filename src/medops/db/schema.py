"""Database schema (SQLAlchemy Core). Same code on SQLite (zero-config), Docker PostgreSQL
and Cloud SQL — only the connection URL changes.

analysis_results : 1 row = 1 case x 1 vertebral level
analysis_runs    : execution metadata (traceability). Weight *names* only, never paths.
"""
from __future__ import annotations

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, MetaData, Numeric, String, Table, Text,
    TypeDecorator, UniqueConstraint, create_engine, event,
)
from sqlalchemy.engine import Engine

metadata = MetaData()


class _Num(TypeDecorator):
    """numeric in the DB, plain float in Python (pandas / JSON friendly).

    SQLite's NUMERIC affinity stores 400.0 as INTEGER 400, so coerce on the way out too.
    """
    impl = Numeric(10, 2, asdecimal=False)
    cache_ok = True

    def process_result_value(self, value, dialect):
        return None if value is None else float(value)


def _num():
    return _Num()


analysis_runs = Table(
    "analysis_runs", metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", String(64), nullable=False, unique=True),
    Column("case_id", Text, nullable=False, index=True),
    Column("levels", Text, nullable=False),                 # "T8,T12"
    Column("status", String(16), nullable=False),           # queued/running/success/error
    Column("stage", String(32)),
    Column("analysis_mode", String(8), nullable=False),     # mock / live
    Column("module_name", Text),
    Column("module_version", Text),
    Column("study_instance_uid", Text),
    Column("series_instance_uid", Text),
    Column("study_date", String(10)),                     # 'YYYY-MM-DD' (DICOM StudyDate)
    Column("modality", String(16)),                       # of the analyzed series, e.g. CT
    Column("aorta_model", Text),
    Column("tl_model", Text),
    Column("slicer_version", Text),
    Column("engine_backend", String(16)),                 # mock / local / docker / cloudrun
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Column("elapsed_s", Float),
    Column("error_message", Text),
)

analysis_results = Table(
    "analysis_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("analysis_runs.run_id", ondelete="CASCADE"),
           nullable=False, index=True),
    Column("case_id", Text, nullable=False, index=True),
    Column("level", String(8), nullable=False),
    Column("aorta_area_mm2", _num()),
    Column("aorta_dia_mm", _num()),
    Column("tl_area_mm2", _num(), nullable=True),
    Column("fl_area_mm2", _num(), nullable=True),
    Column("tl_ratio_pct", _num(), nullable=True),
    UniqueConstraint("run_id", "level", name="uq_results_run_level"),
)


def make_engine(url: str) -> Engine:
    kw = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        # Small pool: the smallest Cloud SQL tier allows ~25 connections in total, shared
        # by Orthanc, the job server and every UI instance.
        kw.update(pool_size=3, max_overflow=2, pool_recycle=1800)
    engine = create_engine(url, **kw)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):  # SQLite needs FK enforcement switched on per connection
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")  # job server writes while the UI reads
            cur.close()
    return engine


def init_db(engine: Engine) -> None:
    if engine.url.get_backend_name() == "sqlite" and engine.url.database:
        from pathlib import Path
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Tiny forward-only migration: add columns introduced after a DB was created
    (create_all never alters existing tables). Additive and nullable only."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as c:
        for table in metadata.sorted_tables:
            existing = {col["name"] for col in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing and col.nullable:
                    ddl = col.type.compile(dialect=engine.dialect)
                    c.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'))
