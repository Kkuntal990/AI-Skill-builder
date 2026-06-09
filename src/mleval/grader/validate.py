"""Format-only submission validator — the MLE-Bench ``validate_submission`` affordance.

WHY THIS EXISTS
---------------
MLE-Bench gives the agent a ``validate_submission.sh`` → ``/validate`` endpoint
that checks a submission is well-formed and returns a validity *message* but
**never a score** (``grading_server.py`` discards the bool and returns only the
message; ``instructions.txt``: *"does NOT give you a score"*). Our MLEvolve
harness runs with ``use_grading_server: false`` and MLEvolve's only internal
signal is the agent's self-reported stdout scalar — so the agent has no way to
learn that its ``submission.csv`` has the wrong columns and games the
self-metric (e.g. emitting a ``label`` column), which the held-out grader then
correctly rejects as INVALID.

This restores that exact affordance: a format/structure check the agent can run
during the search, returning VALID / INVALID + reasons and **never a score**.
It deliberately reuses the held-out grader's *structural gates only* (required
columns, id-set equality, duplicate ids) and never computes or reports the
metric. To avoid any score-leak, it reads ONLY the ``id`` column of the
references (the test-split id-set is public) — never the answer column.

Faithful to MLE-Bench: scoring still happens exactly once, post-run, on the one
self-selected ``best_submission/submission.csv`` (``mleval.grader``). This tool
only tells the agent whether that file is *gradable*, not how it scored.

CLI (agent-invokable; mirrors validate_submission.sh)::

    python -m mleval.grader.validate submission/submission.csv

Task + references default from the ``$TASK`` and ``$MLEVAL_TASK_REFS_PATH`` env
vars the entrypoint exports, so the agent need only pass its submission path.
Prints ``VALID`` (exit 0) or ``INVALID: <reasons>`` (exit 1).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from .grade import _read_csv_column_pair

# Per-task column contract — kept in lockstep with grader.__main__._TASKS.
# (pred_col is the column the agent must emit; ref_col is read for the id-set
# only, never its values.)
_TASK_COLUMNS: dict[str, dict[str, str]] = {
    "samsum": {"id_col": "id", "pred_col": "generated_summary", "ref_col": "reference_summary"},
    "gsm8k": {"id_col": "id", "pred_col": "prediction", "ref_col": "reference_answer"},
}


def _read_id_set(refs_path: Path, id_col: str) -> tuple[set[str], list[str]]:
    """Public id-set from the references' ``id`` column ONLY (never the answers)."""
    if not refs_path.is_file():
        return set(), [f"references not found: {refs_path}"]
    ids: set[str] = set()
    try:
        with refs_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if id_col not in (reader.fieldnames or []):
                return set(), [f"references missing '{id_col}' column"]
            for row in reader:
                rid = (row.get(id_col) or "").strip()
                if rid:
                    ids.add(rid)
    except (csv.Error, UnicodeDecodeError, OSError) as e:
        return set(), [f"failed to read references {refs_path}: {e}"]
    return ids, []


def validate_format(
    predictions_path: str | Path,
    references_path: str | Path,
    *,
    id_col: str,
    pred_col: str,
    max_id_examples: int = 5,
) -> tuple[bool, list[str]]:
    """Structural validity of a submission — NO score, ever.

    Gates mirror ``grade_predictions`` (required columns, id-set equality, no
    duplicate ids) but stop before scoring. Returns ``(valid, errors)``.
    """
    predictions_path = Path(predictions_path)
    references_path = Path(references_path)

    ref_ids, ref_errs = _read_id_set(references_path, id_col)
    if ref_errs or not ref_ids:
        return False, [f"references: {e}" for e in (ref_errs or ["empty references"])]

    preds, pred_errs = _read_csv_column_pair(predictions_path, id_col, pred_col)
    errors = list(pred_errs)

    structural_fault = any(
        ("not found" in e) or ("missing required" in e) for e in pred_errs
    )
    has_dupes = any(e.startswith("duplicate id") for e in pred_errs)

    pred_ids = set(preds)
    missing = ref_ids - pred_ids
    extra = pred_ids - ref_ids
    if missing:
        eg = sorted(missing)[:max_id_examples]
        errors.append(f"missing {len(missing)} required ids (e.g. {eg})")
    if extra:
        eg = sorted(extra)[:max_id_examples]
        errors.append(f"{len(extra)} unexpected ids (e.g. {eg})")

    valid = not (structural_fault or has_dupes or missing or extra)
    return valid, errors


def _default_refs() -> Path | None:
    refs = os.environ.get("MLEVAL_TASK_REFS_PATH")
    return Path(refs) if refs else None


def main(argv: list[str] | None = None) -> int:
    """Validate a submission's FORMAT (no score). Exit 0 if valid, 1 if not."""
    p = argparse.ArgumentParser(
        description="Format-only submission validator (no score) — MLE-Bench-style",
    )
    p.add_argument("submission", type=Path, help="Path to your submission.csv")
    p.add_argument("--task", default=os.environ.get("TASK", ""),
                   help="Task name (defaults to $TASK)")
    p.add_argument("--refs", type=Path, default=_default_refs(),
                   help="References CSV for the id-set (defaults to $MLEVAL_TASK_REFS_PATH)")
    args = p.parse_args(argv)

    cfg = _TASK_COLUMNS.get(args.task)
    if cfg is None:
        known = sorted(_TASK_COLUMNS)
        print(f"INVALID: unknown task '{args.task}' (set --task or $TASK; known: {known})")
        return 1
    if args.refs is None:
        print("INVALID: no references path (pass --refs or set $MLEVAL_TASK_REFS_PATH)")
        return 1

    valid, errors = validate_format(
        args.submission, args.refs, id_col=cfg["id_col"], pred_col=cfg["pred_col"]
    )
    if valid:
        note = f" ({'; '.join(errors)})" if errors else ""
        cols = f"{cfg['id_col']},{cfg['pred_col']}"
        print(f"VALID: submission has columns '{cols}' over the expected id-set{note}")
        print("(format check only — this does NOT report your score)")
        return 0
    print(f"INVALID: {'; '.join(errors) if errors else 'submission failed format checks'}")
    print("(format check only — fix the above; this does NOT report your score)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
