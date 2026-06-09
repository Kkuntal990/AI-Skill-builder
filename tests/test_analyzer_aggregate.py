"""Schema-1.0 regression guard for the analyzer aggregate layer.

The aggregate/hallucination/journal-picker readers silently rotted once the
adapter bumped ``trajectory.jsonl`` to ``schema_version: "1.0"`` — they still
read the pre-1.0 layout (``usage.input_tokens``, ``output.errors``,
``stage.sub_stage``) and crashed on every current run. This test builds a
minimal schema-1.0 trajectory tree by hand and asserts the aggregate report
parses it and computes the planned metric set, so the rot cannot recur
unnoticed. Hand-authored field shapes mirror ``adapter_mlevolve._record``.
"""
from __future__ import annotations

import json
from pathlib import Path

from mleval.analyzer import aggregate


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows))


def _record(step, node_id, parent_id, stage, is_buggy, is_valid, metric,
            exc_type, subs, code, tok_in, tok_out):
    """One schema-1.0 trajectory.jsonl node (mirrors adapter_mlevolve._record)."""
    return {
        "schema_version": "1.0",
        "step": step,
        "node_id": node_id,
        "parent_id": parent_id,
        "stage": stage,
        "is_buggy": is_buggy,
        "is_valid": is_valid,
        "metric": metric,
        "exc_type": exc_type,
        "exec_time_sec": 12.0,
        "ctime": float(step),
        "code": code,
        "llm_total_in_tokens": tok_in,
        "llm_total_out_tokens": tok_out,
        "llm_calls": [],
        "stage_classifier": {
            "sub_stage": subs[0] if subs else "unknown",
            "all_sub_stages": subs,
            "parse_status": "ok",
        },
    }


def _make_traj(root: Path, cell: str, seed: int, held_score, held_valid: bool):
    d = root / f"run-samsum-{cell}-s{seed}"
    _write(d / "manifest.json", {
        "schema_version": "1.0",
        "run_id": "run",
        "trajectory_id": d.name,
        "task": {"name": "samsum"},
        "cell": {"name": cell},
        "seed": seed,
        "agent": {"name": "mlevolve", "llm_model": "deepseek/deepseek-chat"},
        "timestamps": {"wall_clock_sec": 3600},
        "result": {"exit_code": 0, "status": "completed"},
    })
    recs = [
        _record(0, "n0", None, "root", None, None, None, None, [], "", 0, 0),
        _record(1, "n1", "n0", "draft", True, None, None, "TimeoutError",
                ["1a", "3c", "6c"], "import torch\nfrom peft import LoraConfig", 100, 10),
        _record(2, "n2", "n1", "debug", False, True, 0.30,
                None, ["1a", "3c", "6c"], "from peft import LoraConfig", 120, 12),
        _record(3, "n3", "n2", "improve", True, None, None, "NameError",
                ["1a", "4b"], "x = undefined_name", 130, 9),
    ]
    _write_jsonl(d / "trajectory.jsonl", recs)
    # AIDE-style journal the journal-direct metrics read.
    _write(d / "mlevolve_runs/20260101_000000_x/logs/journal.json", {
        "nodes": [
            {"id": "n0", "step": 0, "is_buggy": False, "metric": None, "exec_time": 1.0},
            {"id": "n1", "step": 1, "is_buggy": True, "metric": None, "exec_time": 10.0},
            {"id": "n2", "step": 2, "is_buggy": False,
             "metric": {"value": 0.30, "maximize": True}, "exec_time": 20.0},
            {"id": "n3", "step": 3, "is_buggy": True, "metric": None, "exec_time": 5.0},
        ],
        "node2parent": {"n1": "n0", "n2": "n1", "n3": "n2"},
    })
    _write_jsonl(d / "prompts.jsonl", [
        {"in_tokens": 100, "out_tokens": 10, "req_time_sec": 2.0,
         "model": "deepseek/deepseek-chat", "func_spec_name": None},
    ])
    _write(d / "held_out_score.json", {
        "task": "samsum", "valid": held_valid,
        "score": held_score if held_valid else None,
        "metric": "rougeL_f",
    })
    return d


def test_aggregate_parses_schema_1_0_and_computes_metrics(tmp_path):
    run = tmp_path / "run"
    _make_traj(run, "with_skill", 0, 0.30, True)
    _make_traj(run, "without_skill", 0, 0.20, True)

    report = aggregate.aggregate(run)

    # Parsed both trajectories without crashing (the rot symptom was a crash).
    assert report["trajectory_count"] == 2
    by_cell = {s["cell"]: s for s in report["trajectories"]}

    w = by_cell["with_skill"]
    # Token totals come from the flat schema-1.0 fields (not usage.*).
    assert w["input_tokens"] == 100 + 120 + 130
    assert w["output_tokens"] == 10 + 12 + 9
    # error_nodes counts is_buggy (not output.errors).
    assert w["error_nodes"] == 2
    # MCGS stage counts + the 16-sub-stage multi-label counts both present.
    assert w["stage_counts"]["debug"] == 1
    assert w["sub_stage_counts"]["3c"] == 2  # nodes 1 and 2 touch adapter_config
    # hallucination via exc_type (NameError is a hallucination class).
    assert w["hallucination"]["errored"] == 2
    assert w["hallucination"]["hallucinated"] == 1
    # first valid submission from is_valid (step 2), no working_dirs needed.
    assert w["first_valid_submission"]["step"] == 2
    # predicates computed inline (state.json absent).
    assert w["predicates_total"] >= 1
    # per-sub-stage decomposition is attached.
    assert "3c" in w["stage_metrics"]

    # Paired lift computes: with(0.30) - without(0.20) = 0.10.
    assert report["pair_count"] == 1
    assert abs(report["lift_mean"] - 0.10) < 1e-9
    # chi-square defaults to the 16-sub-stage granularity.
    assert report["stage_chi_square"]["counts_key"] == "sub_stage_counts"


def test_aggregate_drift_run_scores_none(tmp_path):
    """An invalid held-out run contributes no best_metric (drops from lift)."""
    run = tmp_path / "run"
    _make_traj(run, "with_skill", 0, None, False)
    report = aggregate.aggregate(run)
    s = report["trajectories"][0]
    assert s["best_metric"] is None
    assert s["held_out_valid"] is False
