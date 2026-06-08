"""Align a prediction artifact to held-out references and score it.

The validation gates mirror mle-bench's grade functions (id-set equality,
no duplicates, real per-example predictions). A prediction file that fails
any gate yields ``valid=False`` with a specific reason and ``score=None`` —
which is exactly the outcome we want for an off-task / drifted trajectory
(it cannot fake a high held-out number).

CSV parsing uses the stdlib ``csv`` module (no pandas) so the grader has no
heavy dependency and runs anywhere.
"""
from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .rouge import rouge_l_f


@dataclass
class GradeResult:
    """Outcome of grading one prediction artifact against references."""

    valid: bool
    score: float | None
    metric: str
    n_scored: int
    n_expected: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for held_out_score.json)."""
        return asdict(self)


def _read_csv_column_pair(
    path: Path, id_col: str, value_col: str
) -> tuple[dict[str, str], list[str]]:
    """Read a 2-column id→value mapping. Returns (mapping, errors).

    Empty/whitespace values are preserved as "" (a legitimately empty
    prediction scores 0 ROUGE-L; it is NOT treated as a missing row).
    Duplicate ids are reported as an error (caller decides validity).
    """
    errors: list[str] = []
    mapping: dict[str, str] = {}
    if not path.is_file():
        return mapping, [f"file not found: {path}"]
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if id_col not in fieldnames or value_col not in fieldnames:
                return mapping, [
                    f"missing required columns: need '{id_col}' and '{value_col}', "
                    f"got {fieldnames}"
                ]
            seen_dupes: set[str] = set()
            for row in reader:
                rid = (row.get(id_col) or "").strip()
                if rid == "":
                    errors.append("row with empty id")
                    continue
                if rid in mapping and rid not in seen_dupes:
                    seen_dupes.add(rid)
                    errors.append(f"duplicate id: {rid}")
                val = row.get(value_col)
                mapping[rid] = val if isinstance(val, str) else ""
    except (csv.Error, UnicodeDecodeError, OSError) as e:
        return {}, [f"failed to read {path}: {e}"]
    return mapping, errors


def grade_predictions(
    predictions_path: str | Path,
    references_path: str | Path,
    *,
    id_col: str = "id",
    pred_col: str = "generated_summary",
    ref_col: str = "reference_summary",
    metric: str = "rougeL_f",
    scorer: Callable[[str, str], float] | None = None,
    max_id_examples: int = 5,
) -> GradeResult:
    """Grade ``predictions_path`` against ``references_path`` with ROUGE-L F1.

    Validity gates (any failure → valid=False, score=None):
      * references unreadable / missing columns (a harness error, surfaced)
      * predictions unreadable / missing columns
      * duplicate ids in predictions
      * prediction id-set != reference id-set (missing or extra ids)
    """
    predictions_path = Path(predictions_path)
    references_path = Path(references_path)

    refs, ref_errs = _read_csv_column_pair(references_path, id_col, ref_col)
    if ref_errs or not refs:
        return GradeResult(
            valid=False,
            score=None,
            metric=metric,
            n_scored=0,
            n_expected=len(refs),
            errors=[f"references: {e}" for e in (ref_errs or ["empty references"])],
        )

    preds, pred_errs = _read_csv_column_pair(predictions_path, id_col, pred_col)
    n_expected = len(refs)

    errors = list(pred_errs)
    ref_ids = set(refs)
    pred_ids = set(preds)
    missing = ref_ids - pred_ids
    extra = pred_ids - ref_ids
    if missing:
        sample = sorted(missing)[:max_id_examples]
        errors.append(f"missing {len(missing)} predicted ids (e.g. {sample})")
    if extra:
        sample = sorted(extra)[:max_id_examples]
        errors.append(f"{len(extra)} unexpected predicted ids (e.g. {sample})")

    # Invalidate ONLY on structural faults: id-set mismatch (missing/extra),
    # duplicate ids, a missing file, or missing columns. A benign note such as
    # "row with empty id" (a stray malformed row that does NOT change the
    # surviving id-set) is intentionally non-fatal — it cannot inflate the
    # score, since every reference id is still scored (a non-predicted id gets
    # "" → 0.0 ROUGE-L) and the denominator stays len(refs).
    has_dupes = any(e.startswith("duplicate id") for e in pred_errs)
    structural_pred_fault = any(
        ("not found" in e) or ("missing required" in e) for e in pred_errs
    )
    if errors and (missing or extra or has_dupes or structural_pred_fault):
        return GradeResult(
            valid=False,
            score=None,
            metric=metric,
            n_scored=0,
            n_expected=n_expected,
            errors=errors,
        )

    score_fn = scorer or rouge_l_f
    total = 0.0
    for rid, ref_text in refs.items():
        total += score_fn(preds.get(rid, ""), ref_text)
    score = total / n_expected if n_expected else 0.0

    return GradeResult(
        valid=True,
        score=score,
        metric=metric,
        n_scored=n_expected,
        n_expected=n_expected,
        errors=errors,  # non-fatal notes (e.g. stray empty-id rows), if any
    )
