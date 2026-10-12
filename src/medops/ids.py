"""Identifier validation shared by every layer.

case_id is used to build filesystem paths, so it is restricted to ^[A-Za-z0-9_-]+$
(path traversal guard). Levels are vertebral codes like "T8" / "T12".
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
LEVEL_RE = re.compile(r"^[CTL]([1-9]|1[0-2])$")
DEFAULT_LEVELS = ["T8", "T12"]


def validate_case_id(case_id: str) -> str:
    if not isinstance(case_id, str) or not CASE_ID_RE.match(case_id):
        raise ValueError("case_id must match ^[A-Za-z0-9_-]+$")
    return case_id


def normalize_levels(levels: Optional[Iterable[str]]) -> list[str]:
    if not levels:
        return list(DEFAULT_LEVELS)
    out = []
    for lv in levels:
        code = str(lv).strip().upper()
        if not LEVEL_RE.match(code):
            raise ValueError(f"invalid vertebral level: {lv!r} (expected e.g. 'T8', 'T12')")
        if code not in out:
            out.append(code)
    return out
