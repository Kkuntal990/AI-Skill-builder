"""Produce L1 + L2 side-by-side report for spike-012 paired cells.

Usage (inside helper pod):
    PYTHONPATH=/workspace python3 /results/.../l1_l2_compare.py \\
        /results/mlevolve-spike-012/mlevolve-spike-012-samsum-with-skill-s0 \\
        /results/mlevolve-spike-012/mlevolve-spike-012-samsum-without-skill-s0

Reads each trajectory's:
  - trajectory.jsonl (from adapter_mlevolve + stage_classifier)
  - prompts.jsonl (from prompt_logger)
  - mlevolve_runs/*/logs/journal.json

Outputs:
  - markdown table on stdout
  - JSON summary at <run_dir>/_l1_l2_report.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Make repo importable
sys.path.insert(0, "/workspace")

from mleval.analyzer import metrics  # noqa: E402
from mleval.analyzer import stage_metrics as stage_metrics_mod  # noqa: E402

REPO_ROOT = Path("/workspace")


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def find_journal(traj_dir: Path) -> dict:
    matches = list(traj_dir.rglob("journal.json"))
    if not matches:
        return {}
    return json.loads(matches[0].read_text())


def layer1(traj_dir: Path) -> dict:
    """Raw counts from journal + prompts."""
    journal = find_journal(traj_dir)
    nodes = journal.get("nodes", []) or []
    prompts = read_jsonl(traj_dir / "prompts.jsonl")

    metrics_seen = []
    for n in nodes:
        m = n.get("metric")
        v = m.get("value") if isinstance(m, dict) else m
        if v is not None:
            metrics_seen.append(v)

    buggy_chain = 0
    for n in nodes:
        if n.get("stage") == "root":
            continue
        if n.get("is_buggy"):
            buggy_chain += 1
        else:
            break

    exc_taxonomy = dict(Counter(n.get("exc_type") for n in nodes if n.get("exc_type")))

    return {
        "best_metric": max(metrics_seen) if metrics_seen else None,
        "success_count": len(metrics_seen),
        "nodes_total": len(nodes),
        "nodes_buggy": sum(1 for n in nodes if n.get("is_buggy")),
        "nodes_draft": sum(1 for n in nodes if n.get("stage") == "draft"),
        "nodes_debug": sum(1 for n in nodes if n.get("stage") == "debug"),
        "nodes_improve": sum(1 for n in nodes if n.get("stage") == "improve"),
        "consecutive_buggy_chain": buggy_chain,
        "exception_taxonomy": exc_taxonomy,
        "llm_calls": len(prompts),
        "llm_input_tokens_sum": sum((p.get("in_tokens") or 0) for p in prompts),
        "llm_output_tokens_sum": sum((p.get("out_tokens") or 0) for p in prompts),
        "llm_walltime_sec": round(sum((p.get("req_time_sec") or 0) for p in prompts), 1),
        "llm_calls_missing_tokens": sum(1 for p in prompts if p.get("in_tokens") is None),
    }


def layer2(traj_dir: Path) -> dict:
    """Derived metrics + stage distribution (multi-label aware) from classifier."""
    derived = metrics.per_trajectory(traj_dir, REPO_ROOT)

    traj = read_jsonl(traj_dir / "trajectory.jsonl")

    # Primary (single-label) distribution — back-compat, highest-priority match per node
    primary_top_levels = Counter()
    primary_sub_stages = Counter()
    # Multi-label distribution — each node contributes to every stage its code touches
    union_top_levels = Counter()
    union_sub_stages = Counter()
    # Parse status distribution
    parse_status_counts = Counter()
    # Per-node summary
    per_node = []

    for rec in traj:
        sc = rec.get("stage_classifier") or {}
        primary_top_levels[sc.get("top_level", "?")] += 1
        primary_sub_stages[sc.get("sub_stage", "?")] += 1
        parse_status_counts[sc.get("parse_status", "?")] += 1
        for tl in sc.get("all_top_levels") or []:
            union_top_levels[tl] += 1
        for ss in sc.get("all_sub_stages") or []:
            union_sub_stages[ss] += 1
        per_node.append({
            "node_id": (rec.get("node_id") or "")[:8],
            "upstream_stage": rec.get("stage"),
            "primary_label": sc.get("label"),
            "all_sub_stages": sc.get("all_sub_stages") or [],
            "parse_status": sc.get("parse_status"),
            "is_buggy": rec.get("is_buggy"),
            "exc_type": rec.get("exc_type"),
        })

    return {
        "derived_metrics": derived,
        "stage_classifier_distribution": {
            "primary_top_levels": dict(primary_top_levels),
            "primary_sub_stages": dict(primary_sub_stages),
            "union_top_levels": dict(union_top_levels),
            "union_sub_stages": dict(union_sub_stages),
            "parse_status": dict(parse_status_counts),
        },
        "per_node": per_node,
    }


def method_fingerprint(traj_dir: Path) -> dict:
    """L2 supplementary: did each cell use specific methods (parsed from code summaries)."""
    journal = find_journal(traj_dir)
    nodes = journal.get("nodes", []) or []
    summaries = " ".join((n.get("code_summary") or "") for n in nodes).lower()
    return {
        "used_lora": "lora" in summaries,
        "used_qlora_or_4bit": "qlora" in summaries or "4-bit" in summaries or "4bit" in summaries,
        "used_neftune": "neftune" in summaries,
        "used_data_augmentation": "augment" in summaries,
        "used_lmhead_save": "lm_head" in summaries,
    }


def analyze(traj_dir: Path) -> dict:
    return {
        "cell": traj_dir.name.split("samsum-")[1].rsplit("-s", 1)[0],
        "layer1": layer1(traj_dir),
        "layer2": layer2(traj_dir),
        "method_fingerprint": method_fingerprint(traj_dir),
    }


def fmt_val(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    if isinstance(v, dict) and not v:
        return "{}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={v[k]}" for k in sorted(v))[:60]
    return str(v)


def main(argv):
    if len(argv) != 3:
        print("usage: l1_l2_compare.py <with_skill_traj_dir> <without_skill_traj_dir>", file=sys.stderr)
        return 1

    ws = analyze(Path(argv[1]))
    ns = analyze(Path(argv[2]))

    # Layer 1
    print("# L1 + L2 Side-by-Side")
    print()
    print("## Layer 1 — raw counts")
    print()
    print(f"| Metric | with_skill | without_skill |")
    print(f"|---|---|---|")
    l1_keys = [
        ("best_metric", "Best ROUGE-L F1"),
        ("success_count", "Successful nodes"),
        ("nodes_total", "Total nodes (incl root)"),
        ("nodes_buggy", "Buggy nodes"),
        ("nodes_draft", "Draft nodes"),
        ("nodes_debug", "Debug nodes"),
        ("nodes_improve", "Improve nodes"),
        ("consecutive_buggy_chain", "Consecutive-buggy chain depth"),
        ("exception_taxonomy", "Exception taxonomy"),
        ("llm_calls", "LLM calls"),
        ("llm_input_tokens_sum", "Input tokens (sum)"),
        ("llm_output_tokens_sum", "Output tokens (sum)"),
        ("llm_walltime_sec", "LLM wall-time (s)"),
        ("llm_calls_missing_tokens", "LLM calls without token counts"),
    ]
    for key, label in l1_keys:
        a = ws["layer1"].get(key)
        b = ns["layer1"].get(key)
        print(f"| {label} | {fmt_val(a)} | {fmt_val(b)} |")

    # Layer 2 — derived
    print()
    print("## Layer 2 — derived metrics (from metrics.py)")
    print()
    print(f"| Metric | with_skill | without_skill |")
    print(f"|---|---|---|")
    d_ws = ws["layer2"]["derived_metrics"]
    d_ns = ns["layer2"]["derived_metrics"]
    l2_keys = [
        ("cost_usd", "Cost (USD)"),
        ("llm_call_count", "LLM call count"),
        ("step_count", "Search steps"),
        ("redundant_loops", "Redundant loops"),
        ("self_correction_rate", "Self-correction rate"),
    ]
    for key, label in l2_keys:
        print(f"| {label} | {fmt_val(d_ws.get(key))} | {fmt_val(d_ns.get(key))} |")

    # Nested LLM latency
    lat_ws = d_ws.get("llm_latency") or {}
    lat_ns = d_ns.get("llm_latency") or {}
    print(f"| LLM latency p50 | {fmt_val(lat_ws.get('p50'))} | {fmt_val(lat_ns.get('p50'))} |")
    print(f"| LLM latency p95 | {fmt_val(lat_ws.get('p95'))} | {fmt_val(lat_ns.get('p95'))} |")

    # Step exec time
    sx_ws = d_ws.get("step_exec_time") or {}
    sx_ns = d_ns.get("step_exec_time") or {}
    print(f"| Step exec p50 (s) | {fmt_val(sx_ws.get('p50'))} | {fmt_val(sx_ns.get('p50'))} |")
    print(f"| Step exec max (s) | {fmt_val(sx_ws.get('max'))} | {fmt_val(sx_ns.get('max'))} |")

    # Hallucination
    h_ws = d_ws.get("hallucination") or {}
    h_ns = d_ns.get("hallucination") or {}
    print(f"| Hallucination matches | {fmt_val(h_ws.get('count'))} | {fmt_val(h_ns.get('count'))} |")
    print(f"| Hallucination rate | {fmt_val(h_ws.get('rate'))} | {fmt_val(h_ns.get('rate'))} |")

    # First valid submission (= time to first success)
    fvs_ws = d_ws.get("first_valid_submission") or {}
    fvs_ns = d_ns.get("first_valid_submission") or {}
    print(f"| Time to first success (s) | {fmt_val(fvs_ws.get('time_sec') if isinstance(fvs_ws, dict) else fvs_ws)} | {fmt_val(fvs_ns.get('time_sec') if isinstance(fvs_ns, dict) else fvs_ns)} |")
    print(f"| Steps to first success | {fmt_val(fvs_ws.get('step') if isinstance(fvs_ws, dict) else None)} | {fmt_val(fvs_ns.get('step') if isinstance(fvs_ns, dict) else None)} |")

    # Skill API adoption
    sa_ws = d_ws.get("skill_api_adoption") or {}
    sa_ns = d_ns.get("skill_api_adoption") or {}
    print(f"| Skill API adoption (with_skill only) | {fmt_val(sa_ws.get('adopted_apis') if isinstance(sa_ws, dict) else None)} | n/a |")

    # Layer 2 — stage distribution (multi-label)
    print()
    print("## Layer 2 — parse status (parse_error = diff-patch corrupted code)")
    print()
    ps_ws = ws["layer2"]["stage_classifier_distribution"]["parse_status"]
    ps_ns = ns["layer2"]["stage_classifier_distribution"]["parse_status"]
    print(f"| Parse status | with_skill | without_skill |")
    print(f"|---|---|---|")
    for k in sorted(set(ps_ws) | set(ps_ns)):
        print(f"| {k} | {ps_ws.get(k, 0)} | {ps_ns.get(k, 0)} |")

    print()
    print("## Layer 2 — stage coverage (union of ALL matching rules, multi-label)")
    print()
    union_ws = ws["layer2"]["stage_classifier_distribution"]["union_sub_stages"]
    union_ns = ns["layer2"]["stage_classifier_distribution"]["union_sub_stages"]
    print(f"| Sub-stage | with_skill | without_skill |")
    print(f"|---|---|---|")
    for k in sorted(set(union_ws) | set(union_ns)):
        print(f"| {k} | {union_ws.get(k, 0)} | {union_ns.get(k, 0)} |")

    print()
    print("## Layer 2 — per-node fingerprint")
    print()
    print(f"### with_skill")
    print(f"| Node | Upstream stage | Sub-stages touched | Parse | Buggy | Exception |")
    print(f"|---|---|---|---|---|---|")
    for n in ws["layer2"]["per_node"]:
        print(f"| {n['node_id']} | {n['upstream_stage']} | {','.join(n['all_sub_stages']) or '—'} | {n['parse_status']} | {n['is_buggy']} | {n['exc_type'] or '—'} |")
    print()
    print(f"### without_skill")
    print(f"| Node | Upstream stage | Sub-stages touched | Parse | Buggy | Exception |")
    print(f"|---|---|---|---|---|---|")
    for n in ns["layer2"]["per_node"]:
        print(f"| {n['node_id']} | {n['upstream_stage']} | {','.join(n['all_sub_stages']) or '—'} | {n['parse_status']} | {n['is_buggy']} | {n['exc_type'] or '—'} |")

    # Method fingerprint
    print()
    print("## L2 supplementary — method fingerprint (from code_summary text matches)")
    print()
    print(f"| Method | with_skill | without_skill |")
    print(f"|---|---|---|")
    for k in ws["method_fingerprint"]:
        print(f"| {k.replace('_',' ')} | {fmt_val(ws['method_fingerprint'][k])} | {fmt_val(ns['method_fingerprint'][k])} |")

    # Per-sub-stage metrics (clean-reach · rework · failure-modes)
    print()
    stage_metrics_mod.print_paired(Path(argv[1]), Path(argv[2]))

    # Save JSON
    run_dir = Path(argv[1]).parent
    out = run_dir / "_l1_l2_report.json"
    sm = {
        "with_skill": stage_metrics_mod.stage_metrics(Path(argv[1])),
        "without_skill": stage_metrics_mod.stage_metrics(Path(argv[2])),
    }
    with out.open("w") as f:
        json.dump(
            {"with_skill": ws, "without_skill": ns, "stage_metrics": sm},
            f, indent=2, default=str,
        )
    print()
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
