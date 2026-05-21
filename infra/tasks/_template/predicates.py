"""Task-specific state predicates.

Each predicate takes the trajectory's $MLEVAL_OUTPUT_DIR and returns bool.
Loaded by ``mleval.analyzer.state_predicates`` via file-path import.

Add task-specific assertions here. Generic ones (has_best_solution,
metric_finite, etc.) are applied automatically — don't duplicate them.
"""

from __future__ import annotations

from pathlib import Path


def submission_csv_present(output_dir: Path) -> bool:
    """AIDE writes submission to working_dir; we snapshot per-step into working_dirs/."""
    return any(output_dir.glob("working_dirs/*/submission.csv"))


def submission_has_correct_columns(output_dir: Path) -> bool:
    """Sample check — first submission row matches expected schema."""
    import csv

    for path in output_dir.glob("working_dirs/*/submission.csv"):
        try:
            with path.open() as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and {"id", "prediction"}.issubset({c.strip() for c in header}):
                    return True
        except Exception:  # noqa: BLE001
            continue
    return False


PREDICATES = {
    "submission_csv_present": submission_csv_present,
    "submission_has_correct_columns": submission_has_correct_columns,
}
