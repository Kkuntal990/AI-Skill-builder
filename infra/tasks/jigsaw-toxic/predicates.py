"""Task-specific predicates for jigsaw-toxic.

Loaded by ``mleval.analyzer.state_predicates`` via file-path import. Each
predicate takes ``$MLEVAL_OUTPUT_DIR`` and returns bool.
"""

from __future__ import annotations

import csv
from pathlib import Path

CLASSES = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
ID_COL = "id"
REQUIRED_COLS = {ID_COL, *CLASSES}


def _find_submissions(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("working_dirs/*/working/submission.csv"))


def submission_csv_present(output_dir: Path) -> bool:
    return len(_find_submissions(output_dir)) > 0


def submission_has_correct_columns(output_dir: Path) -> bool:
    """Any submission.csv has id + all six toxicity columns."""
    for p in _find_submissions(output_dir):
        try:
            with p.open() as f:
                header = next(csv.reader(f), None)
            if header and REQUIRED_COLS.issubset({c.strip() for c in header}):
                return True
        except OSError:
            continue
    return False


def submission_row_count_matches_test(output_dir: Path) -> bool:
    """Submission has the same row count as test.csv (~153k)."""
    import os

    data_dir = Path(os.environ.get("MLEVAL_TASK_DATA_DIR", ""))
    test_path = data_dir / "test.csv"
    if not test_path.is_file():
        return False
    try:
        with test_path.open() as f:
            expected = sum(1 for _ in f) - 1
    except OSError:
        return False

    for p in _find_submissions(output_dir):
        try:
            with p.open() as f:
                actual = sum(1 for _ in f) - 1
            if actual == expected:
                return True
        except OSError:
            continue
    return False


def probabilities_in_unit_interval(output_dir: Path) -> bool:
    """All six prob columns ∈ [0, 1] in some submission. ROC AUC is fine with
    any monotone score, but values outside [0,1] usually mean the agent forgot
    to apply a sigmoid — surfacing it is useful."""
    for p in _find_submissions(output_dir):
        try:
            with p.open() as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or not REQUIRED_COLS.issubset(reader.fieldnames):
                    continue
                ok = True
                for row in reader:
                    for c in CLASSES:
                        try:
                            v = float(row[c])
                        except (TypeError, ValueError):
                            ok = False
                            break
                        if not (0.0 <= v <= 1.0):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return True
        except OSError:
            continue
    return False


def predictions_not_constant(output_dir: Path) -> bool:
    """Predictions vary across rows (catches sample-submission stubs of all 0.5).

    Computed on the LAST submission only — earlier ones may legitimately be
    placeholder writes during the agent's iteration."""
    subs = _find_submissions(output_dir)
    if not subs:
        return False
    p = subs[-1]
    try:
        with p.open() as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not REQUIRED_COLS.issubset(reader.fieldnames):
                return False
            seen: dict[str, float] = {}
            for row in reader:
                for c in CLASSES:
                    try:
                        v = float(row[c])
                    except (TypeError, ValueError):
                        return False
                    if c not in seen:
                        seen[c] = v
                    elif seen[c] != v:
                        return True
            return False
    except OSError:
        return False


PREDICATES = {
    "submission_csv_present": submission_csv_present,
    "submission_has_correct_columns": submission_has_correct_columns,
    "submission_row_count_matches_test": submission_row_count_matches_test,
    "probabilities_in_unit_interval": probabilities_in_unit_interval,
    "predictions_not_constant": predictions_not_constant,
}
