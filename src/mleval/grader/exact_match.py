"""Numeric exact-match scorer (GSM8K and similar final-number tasks).

Self-contained and deterministic, mirroring ``rouge.py``: no dependencies, so
the grader stays zero-dep and unit-testable with hand-computed values.

The agent's ``prediction`` column is meant to be a final integer, but models
are messy — they emit ``"42"``, ``"42.0"``, ``"$1,234"``, ``"#### 42"``, or
``"The answer is 42."``. We normalise both sides to a canonical number and
compare as numbers (not strings). An unparseable prediction scores 0.0 rather
than raising, so one bad row never aborts grading.
"""
from __future__ import annotations

import re

# Last number-like token: optional sign, digits with optional thousands commas,
# optional decimal part. We take the LAST match (the final answer).
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_number(s: str) -> float | None:
    """Extract the final numeric value from a string, or None if none found.

    Prefers the token after the last ``####`` cue (GSM8K convention); else the
    last number-like token anywhere in the string. Strips ``$``, thousands
    commas, and whitespace. Returns a float (callers compare with ==).
    """
    if s is None:
        return None
    text = str(s)
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    text = text.replace("$", "").replace(",", "")
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def exact_match(pred: str, ref: str) -> float:
    """1.0 if pred and ref denote the same number, else 0.0.

    Both are normalised via :func:`to_number`. A missing/unparseable prediction
    (None) is wrong (0.0). Integer-valued floats compare equal regardless of a
    trailing ``.0`` (``42`` == ``42.0``).
    """
    p = to_number(pred)
    r = to_number(ref)
    if p is None or r is None:
        return 0.0
    return 1.0 if p == r else 0.0
