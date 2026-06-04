"""CLI: grade the best node's preserved prediction artifact, post-run.

Invoked by the trajectory entrypoint AFTER MLEvolve exits:

    python3 -m mleval.grader <MLEVAL_OUTPUT_DIR> \
        --task samsum --refs /results/data/samsum/refs/test_refs.csv

It locates ``mlevolve_runs/<ts>/workspace/best_submission/submission.csv``
(the predictions MLEvolve preserved for the best node, because we run with
``no_submission_mode: False``), grades it against the held-out references,
and writes ``held_out_score.json`` into the output dir. This is the
trustworthy A/B metric — distinct from MLEvolve's self-reported journal
metric, which stays the (gameable) tree-search signal.

By contract this NEVER raises and always exits 0: a missing artifact or a
read error becomes ``{"valid": false, "score": null, "error": ...}`` so a
grading hiccup cannot abort the entrypoint's manifest/analyzer chain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .grade import GradeResult, grade_predictions

# Per-task grading config. Each task names its prediction/reference columns
# and the metric. GSM8K (exact-match) and BoolQ (accuracy) become siblings.
_TASKS: dict[str, dict[str, str]] = {
    "samsum": {
        "id_col": "id",
        "pred_col": "generated_summary",
        "ref_col": "reference_summary",
        "metric": "rougeL_f",
    },
}


def _safe_mtime(p: Path) -> float:
    """mtime, or 0.0 for a broken symlink / vanished file (never raises)."""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _find_best_submission(output_dir: Path) -> Path | None:
    """Newest ``best_submission/submission.csv`` under the run tree, if any."""
    runs_dir = output_dir / "mlevolve_runs"
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        runs_dir.glob("*/workspace/best_submission/submission.csv"),
        key=_safe_mtime,
    )
    return candidates[-1] if candidates else None


def _write(out_path: Path, payload: dict) -> None:
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run(
    output_dir: Path,
    task: str,
    refs_path: Path,
    out_path: Path,
    pred_path: Path | None = None,
) -> dict:
    """Grade and return the payload dict (also written to ``out_path``)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = {"task": task, "source": "held_out_grader", "metric": _TASKS.get(task, {}).get("metric")}

    cfg = _TASKS.get(task)
    if cfg is None:
        payload = {**base, "valid": False, "score": None, "error": f"unknown task: {task}"}
        _write(out_path, payload)
        return payload

    if pred_path is None:
        pred_path = _find_best_submission(output_dir)
    if pred_path is None:
        payload = {
            **base,
            "valid": False,
            "score": None,
            "error": "no best_submission/submission.csv found (agent produced no valid artifact)",
        }
        _write(out_path, payload)
        return payload

    if not refs_path.is_file():
        payload = {
            **base,
            "valid": False,
            "score": None,
            "error": f"references not found: {refs_path}",
        }
        _write(out_path, payload)
        return payload

    try:
        result: GradeResult = grade_predictions(
            pred_path,
            refs_path,
            id_col=cfg["id_col"],
            pred_col=cfg["pred_col"],
            ref_col=cfg["ref_col"],
            metric=cfg["metric"],
        )
        payload = {
            **base,
            **result.to_dict(),
            "predictions_path": str(pred_path),
            "references_path": str(refs_path),
        }
    except Exception as e:
        payload = {**base, "valid": False, "score": None, "error": f"grader exception: {e!r}"}

    _write(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    """Parse args, grade, write held_out_score.json. Always returns 0."""
    p = argparse.ArgumentParser(description="Held-out grader for agent prediction artifacts")
    p.add_argument("output_dir", type=Path, help="MLEVAL_OUTPUT_DIR for this trajectory")
    p.add_argument("--task", required=True, help="Task name (e.g. samsum)")
    p.add_argument("--refs", required=True, type=Path, help="Held-out references CSV")
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON (default: <output_dir>/held_out_score.json)",
    )
    p.add_argument(
        "--pred", type=Path, default=None,
        help="Override predictions CSV path (else auto-located)",
    )
    args = p.parse_args(argv)

    out_path = args.out or (args.output_dir / "held_out_score.json")
    # Hard never-raise contract: a grading hiccup (disk full, bad path, broken
    # symlink) must still leave a held_out_score.json and exit 0, so it cannot
    # abort the entrypoint's manifest/analyzer chain.
    try:
        payload = run(args.output_dir, args.task, args.refs, out_path, pred_path=args.pred)
    except Exception as e:
        payload = {
            "task": args.task,
            "source": "held_out_grader",
            "valid": False,
            "score": None,
            "error": f"grader top-level exception: {e!r}",
        }
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except OSError:
            pass

    status = "valid" if payload.get("valid") else "INVALID"
    score = payload.get("score")
    score_str = f"{score:.5f}" if isinstance(score, (int, float)) else "-"
    err = payload.get("error") or "; ".join(payload.get("errors", []))
    err_str = f"({err})" if err else ""
    print(f"[grader] {args.task}: {status} score={score_str} {err_str} -> {out_path}")
    return 0  # never fail the entrypoint chain


if __name__ == "__main__":
    sys.exit(main())
