"""Convert an AIDE trajectory output into our universal trajectory.jsonl.

AIDE writes one ``journal.json`` per run with N nodes; our sidecar writes
``prompts.jsonl`` with M LLM-call records. This adapter joins them and emits
``trajectory.jsonl`` (one line per node) matching the schema in
``infra/agents/_interface.md``.

Join rule (AIDE-specific):
    AIDE makes EXACTLY 2 LLM calls per step under steady state:
      1. code generation     (func_spec is None)
      2. judge / review      (func_spec == "submit_review")
    The parse-retry path in plan_and_code_query may add 1-2 extra calls
    *before* the judge call. We group prompts into per-step bundles by
    walking the list and closing a bundle at each "submit_review" record.

Stage labels are populated by ``stage_classifier`` in a separate pass; this
adapter sets ``stage = unknown / 0.0`` for every record.

CLI:
    python -m mleval.analyzer.adapter_aide $MLEVAL_OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _group_prompts_by_step(prompts: list[dict]) -> list[list[dict]]:
    """Walk prompts left-to-right, close a bundle at each 'submit_review' call."""
    bundles: list[list[dict]] = []
    current: list[dict] = []
    for p in prompts:
        current.append(p)
        if p.get("func_spec_name") == "submit_review":
            bundles.append(current)
            current = []
    if current:
        # Trailing bundle without a judge — usually means trajectory crashed
        # mid-step. Keep it so the trace is faithful.
        bundles.append(current)
    return bundles


def _operator_native(node: dict, node2parent: dict[str, str | None]) -> str:
    """Map AIDE's draft/debug/improve classification by walking the parent edge."""
    parent_id = node2parent.get(node["id"])
    if parent_id is None:
        return "draft"
    # AIDE's stage_name property reads cfg from node, which we don't have here,
    # but the policy is: if parent is buggy -> debug, else -> improve.
    # We can't easily look up the parent buggy flag without indexing first;
    # caller passes the full index.
    return "improve_or_debug"


def _build_record(
    *,
    record_id: str,
    run_id: str,
    trajectory_id: str,
    agent_name: str,
    agent_version: str,
    node: dict,
    parent_node: dict | None,
    prompt_bundle: list[dict],
    code_path: str | None,
) -> dict:
    code = node.get("code") or ""
    plan = node.get("plan") or ""
    term_out = node.get("_term_out") or []
    exc_type = node.get("exc_type")
    exc_info = node.get("exc_info")
    exc_stack = node.get("exc_stack")
    metric = node.get("metric")
    is_buggy = node.get("is_buggy")

    in_tokens = sum(p.get("in_tokens", 0) or 0 for p in prompt_bundle)
    out_tokens = sum(p.get("out_tokens", 0) or 0 for p in prompt_bundle)
    llm_req_time = sum(p.get("req_time_sec", 0.0) or 0.0 for p in prompt_bundle)

    if parent_node is None:
        operator_native = "draft"
    elif parent_node.get("is_buggy"):
        operator_native = "debug"
    else:
        operator_native = "improve"

    started_ts = node.get("ctime") or 0.0
    exec_time = float(node.get("exec_time") or 0.0)
    ended_ts = started_ts + exec_time + llm_req_time

    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "operator_native": operator_native,
        },
        "stage": {
            "top_level": "0",
            "sub_stage": "unknown",
            "label": "unknown",
            "classifier_source": "unknown",
            "classifier_confidence": 0.0,
        },
        "code": {
            "emitted_path": code_path,
            "emitted_lines": len(code.splitlines()) if code else 0,
            "imports_top": [],  # filled by stage_classifier
        },
        "execution": {
            "ran": exc_type is None and bool(code),
            "exit_code": 1 if exc_type else 0,
            "wall_clock_sec": exec_time,
            "stdout_tail_sha": None,
            "stderr_tail_sha": None,
        },
        "usage": {
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "wall_clock_sec": llm_req_time + exec_time,
        },
        "state_snapshot": {
            "playground_files": [],
            "predicate_results": {},  # filled by state_predicates
        },
        "timestamp": {
            "started_at": started_ts,
            "ended_at": ended_ts,
        },
        "input": {
            "prompt_hash": None,
            "predecessor_record_id": (
                f"op_{(parent_node['step']):03d}" if parent_node else None
            ),
        },
        "output": {
            "completion_hash": None,
            "errors": (
                [f"{exc_type}: {exc_info}"] if exc_type else []
            ),
        },
        "aide_native": {
            "node_id": node.get("id"),
            "step": node.get("step"),
            "plan": plan,
            "analysis": node.get("analysis"),
            "metric": metric,
            "is_buggy": is_buggy,
            "term_out_lines": len(term_out),
        },
    }


def _find_journal(output_dir: Path) -> Path:
    matches = list(output_dir.rglob("journal.json"))
    if not matches:
        raise FileNotFoundError(
            f"No journal.json under {output_dir} — did AIDE complete?"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple journal.json under {output_dir}: {matches}")
    return matches[0]


def adapt(output_dir: Path) -> Path:
    """Read AIDE outputs from ``output_dir`` and write trajectory.jsonl + code/."""
    journal_path = _find_journal(output_dir)
    journal: dict[str, Any] = json.loads(journal_path.read_text())
    prompts = _load_jsonl(output_dir / "prompts.jsonl")
    manifest_path = output_dir / "manifest.json"
    manifest: dict = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    )

    run_id = manifest.get("run_id", "unknown")
    trajectory_id = manifest.get("trajectory_id", "unknown")
    agent_name = manifest.get("agent", {}).get("name", "aide")
    agent_version = manifest.get("agent", {}).get("version", "main")

    nodes: list[dict] = journal.get("nodes", [])
    node2parent: dict[str, str | None] = journal.get("node2parent", {})
    nodes_by_id = {n["id"]: n for n in nodes}

    # Sort by step so record_id = op_{step:03d} is sequential.
    nodes_sorted = sorted(nodes, key=lambda n: n.get("step", 0))
    bundles = _group_prompts_by_step(prompts)

    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)
    out_path = output_dir / "trajectory.jsonl"

    with out_path.open("w") as fout:
        for i, node in enumerate(nodes_sorted):
            step = int(node.get("step", i))
            record_id = f"op_{step:03d}"
            parent_id = node2parent.get(node["id"])
            parent_node = nodes_by_id.get(parent_id) if parent_id else None

            code_text = node.get("code") or ""
            code_path: str | None
            if code_text:
                code_file = code_dir / f"{record_id}.py"
                code_file.write_text(code_text)
                code_path = f"code/{record_id}.py"
            else:
                code_path = None

            bundle = bundles[i] if i < len(bundles) else []

            rec = _build_record(
                record_id=record_id,
                run_id=run_id,
                trajectory_id=trajectory_id,
                agent_name=agent_name,
                agent_version=agent_version,
                node=node,
                parent_node=parent_node,
                prompt_bundle=bundle,
                code_path=code_path,
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adapt AIDE outputs to trajectory.jsonl",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="$MLEVAL_OUTPUT_DIR containing aide_logs/, prompts.jsonl, manifest.json",
    )
    args = parser.parse_args(argv)
    out = adapt(args.output_dir)
    print(f"[adapter_aide] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
