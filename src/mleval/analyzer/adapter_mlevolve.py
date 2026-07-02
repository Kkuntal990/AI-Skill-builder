"""MLEvolve journal.json + prompts.jsonl → universal trajectory.jsonl.

Reads MLEvolve's per-run artifacts and emits the universal Layer-2 schema
documented in ``infra/agents/_interface.md``. Designed to be small
(spike criterion C4 caps it at 150 LoC); future extensions (parent edges,
metric histories, fusion lineage) can grow without breaking the schema.

Inputs (under MLEVAL_OUTPUT_DIR):
    mlevolve_runs/<ts>_<exp>/logs/journal.json   — MLEvolve search graph
    mlevolve_runs/<ts>_<exp>/logs/metric.txt     — best final metric (optional)
    prompts.jsonl                                 — per LLM call (our sidecar)

Output:
    trajectory.jsonl                         — one record per node, in
                                                 step-number order

Schema per line (universal):
    schema_version, step, node_id, parent_id, code, stage, is_buggy,
    metric, exc_type, exec_time_sec, ctime, llm_calls

Key mapping decisions:
    - MLEvolve uses 7 stages (root, draft, improve, debug, fusion_draft,
      evolution, fusion) — we pass them through as-is. The stage_classifier
      (AST-based) is the authoritative agent-agnostic classifier; this
      adapter just preserves what MLEvolve emitted.
    - llm_calls per node: MLEvolve doesn't tag prompts with node_id, so we
      bucket by time window between the parent node's finish_time and this
      node's finish_time. Best-effort heuristic keyed on the node ctime
      boundaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


def _read_journal(runs_dir: Path) -> tuple[dict[str, Any], Path] | None:
    if not runs_dir.is_dir():
        return None
    # MLEvolve writes runs/<ts>_<exp>/logs/journal.json; we pick the newest.
    # (Path verified empirically on commit 94854f9 spike — the upstream's
    # run.py creates the logs/ subdir before writing.)
    candidates = sorted(runs_dir.glob("*/logs/journal.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    path = candidates[-1]
    return json.loads(path.read_text()), path


def _read_prompts(out_dir: Path) -> list[dict[str, Any]]:
    fp = out_dir / "prompts.jsonl"
    if not fp.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in fp.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _bucket_prompts(prompts: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group prompts by node_id using ctime windows.

    Each node's prompts are those with `ts` in [parent.ctime, node.ctime].
    Nodes without parents (root/draft) own all earlier prompts.
    """
    buckets: dict[str, list[dict[str, Any]]] = {n["id"]: [] for n in nodes}
    if not prompts or not nodes:
        return buckets
    # Index by id for parent lookup
    id_to_node = {n["id"]: n for n in nodes}
    # Sort nodes by ctime so we know each prompt's right-edge
    sorted_nodes = sorted(nodes, key=lambda n: n.get("ctime") or 0)
    for prompt in prompts:
        ts = prompt.get("ts")
        if ts is None:
            continue
        # Find the earliest node whose ctime >= ts AND whose parent.ctime <= ts
        assigned = None
        for n in sorted_nodes:
            node_ct = n.get("ctime") or 0
            if node_ct < ts:
                continue
            parent_id = (n.get("parent") or {}).get("id") if isinstance(n.get("parent"), dict) else None
            parent_ct = (id_to_node.get(parent_id, {}).get("ctime") or 0) if parent_id else 0
            if parent_ct <= ts <= node_ct:
                assigned = n
                break
        if assigned:
            buckets[assigned["id"]].append(prompt)
    return buckets


def _record(node: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    metric = node.get("metric")
    metric_value = None
    if isinstance(metric, dict):
        metric_value = metric.get("value")
    elif isinstance(metric, (int, float)):
        metric_value = metric

    parent = node.get("parent")
    parent_id = parent.get("id") if isinstance(parent, dict) else None

    total_in = sum((p.get("in_tokens") or 0) for p in prompts)
    total_out = sum((p.get("out_tokens") or 0) for p in prompts)
    llm_calls = [
        {
            "func_spec_name": p.get("func_spec_name"),
            "model": p.get("model"),
            "in_tokens": p.get("in_tokens"),
            "out_tokens": p.get("out_tokens"),
            "req_time_sec": p.get("req_time_sec"),
        }
        for p in prompts
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "step": node.get("step"),
        "node_id": node.get("id"),
        "parent_id": parent_id,
        "stage": node.get("stage"),
        "is_buggy": node.get("is_buggy"),
        "is_valid": node.get("is_valid"),
        "metric": metric_value,
        "exc_type": node.get("exc_type"),
        "exec_time_sec": node.get("exec_time"),
        "ctime": node.get("ctime"),
        "code": node.get("code"),
        "plan": node.get("plan"),
        "analysis": node.get("analysis"),
        "llm_total_in_tokens": total_in,
        "llm_total_out_tokens": total_out,
        "llm_calls": llm_calls,
    }


def adapt(out_dir: Path) -> Path:
    runs_dir = out_dir / "mlevolve_runs"
    j = _read_journal(runs_dir)
    if j is None:
        raise FileNotFoundError(f"no journal.json under {runs_dir}")
    journal, journal_path = j

    nodes = journal.get("nodes") or []
    # Sort by step (None → 0) for stable output
    nodes = sorted(nodes, key=lambda n: (n.get("step") or 0, n.get("ctime") or 0))

    prompts = _read_prompts(out_dir)
    buckets = _bucket_prompts(prompts, nodes)

    out_path = out_dir / "trajectory.jsonl"
    with out_path.open("w") as f:
        for n in nodes:
            rec = _record(n, buckets.get(n["id"], []))
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Adapt MLEvolve artifacts to universal trajectory.jsonl")
    p.add_argument("output_dir", type=Path, help="MLEVAL_OUTPUT_DIR for this trajectory")
    args = p.parse_args(argv)

    try:
        out = adapt(args.output_dir)
    except FileNotFoundError as e:
        print(f"[adapter_mlevolve] {e}", file=sys.stderr)
        return 1
    print(f"[adapter_mlevolve] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
