"""Task-specific predicates for house-prices.

Loaded by ``mleval.analyzer.state_predicates`` via file-path import. Each
predicate takes ``$MLEVAL_OUTPUT_DIR`` and returns bool.
"""

from __future__ import annotations

import csv
from pathlib import Path


def _find_submissions(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("working_dirs/*/submission.csv"))


def submission_csv_present(output_dir: Path) -> bool:
    """At least one per-step working_dir snapshot has a submission.csv."""
    return len(_find_submissions(output_dir)) > 0


def submission_has_correct_columns(output_dir: Path) -> bool:
    """Any submission.csv has the required Id,SalePrice columns."""
    for p in _find_submissions(output_dir):
        try:
            with p.open() as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue
                cols = {c.strip() for c in header}
                if {"Id", "SalePrice"}.issubset(cols):
                    return True
        except Exception:  # noqa: BLE001
            continue
    return False


def submission_row_count_matches_test(output_dir: Path) -> bool:
    """Best-effort: submission.csv has ~1459 rows (matches test.csv size)."""
    # Read the staged test.csv from the task data dir to get the expected count.
    import os

    data_dir = Path(os.environ.get("MLEVAL_TASK_DATA_DIR", ""))
    test_path = data_dir / "test.csv"
    if not test_path.is_file():
        return False
    try:
        with test_path.open() as f:
            expected = sum(1 for _ in f) - 1  # minus header
    except Exception:  # noqa: BLE001
        return False

    for p in _find_submissions(output_dir):
        try:
            with p.open() as f:
                actual = sum(1 for _ in f) - 1
            if actual == expected:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def predictions_are_positive(output_dir: Path) -> bool:
    """All SalePrice predictions in any submission.csv are > 0."""
    for p in _find_submissions(output_dir):
        try:
            with p.open() as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "SalePrice" not in reader.fieldnames:
                    continue
                all_positive = all(
                    float(row["SalePrice"]) > 0 for row in reader if row.get("SalePrice")
                )
                if all_positive:
                    return True
        except (ValueError, KeyError):
            continue
        except Exception:  # noqa: BLE001
            continue
    return False


PREDICATES = {
    "submission_csv_present": submission_csv_present,
    "submission_has_correct_columns": submission_has_correct_columns,
    "submission_row_count_matches_test": submission_row_count_matches_test,
    "predictions_are_positive": predictions_are_positive,
}
