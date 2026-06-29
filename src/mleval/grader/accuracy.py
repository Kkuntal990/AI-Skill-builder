"""Boolean accuracy scorer (BoolQ yes/no and similar binary tasks).

Self-contained and deterministic, mirroring ``exact_match.py`` / ``rouge.py``:
no dependencies, so the grader stays zero-dep and unit-testable with
hand-computed values.

The agent's ``prediction`` column is meant to be a yes/no answer, but models
are messy — they emit ``"yes"``, ``"No."``, ``"true"``, ``"1"``, or
``"yes, because the passage says..."``. We normalise both sides to a canonical
boolean (taking the FIRST word-like token) and compare. An unparseable/empty
prediction is wrong (0.0) rather than raising, so one bad row never aborts
grading.
"""
from __future__ import annotations

import re

# First word-like token (models prefix with the answer: "Yes.", "no, because").
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_TRUE = {"true", "yes", "y", "1", "t"}
_FALSE = {"false", "no", "n", "0", "f"}


def to_bool(s: str) -> bool | None:
    """Map a messy yes/no/true/false string to a bool, or None if unparseable.

    Reads the FIRST word-like token (lower-cased), so trailing prose/punctuation
    ("Yes.", "no — the passage...") is tolerated. Returns None for empty or
    non-boolean tokens; callers treat None as wrong.
    """
    if s is None:
        return None
    m = _TOKEN_RE.search(str(s).strip().lower())
    if not m:
        return None
    tok = m.group(0)
    if tok in _TRUE:
        return True
    if tok in _FALSE:
        return False
    return None


def accuracy(pred: str, ref: str) -> float:
    """1.0 if pred and ref denote the same boolean, else 0.0.

    Both are normalised via :func:`to_bool`. A missing/unparseable prediction
    (None) is wrong (0.0). The reference is written canonically by the task's
    make_grading_data.py (``true``/``false``), so it always parses.
    """
    p = to_bool(pred)
    r = to_bool(ref)
    if p is None or r is None:
        return 0.0
    return 1.0 if p == r else 0.0
