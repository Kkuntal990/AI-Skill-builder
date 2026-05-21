"""Aggregate per-trajectory outputs into a sweep-level report.

Walks ``/results/<run_id>/<trajectory_id>/`` for each (task × cell × seed)
combination, reads each ``manifest.json`` + ``trajectory.jsonl`` + best
metric from AIDE's journal, then computes:

    L1 outcome     — mean metric per cell, paired Lift (with - without) per
                     (task, seed), with-vs-without delta + 95% CI.
    L3 cost        — sum tokens, max wall_clock, error count per trajectory.
    Stage activity — count of nodes per sub_stage per cell.

Output is a single ``report.json`` plus a markdown ``report.md`` summary.

This runs *locally* (Mac), after the orchestrator pulls results off the PVC.
The orchestrator's ``--pull-results`` flag handles the kubectl-cp step.

CLI:
    python -m mleval.analyzer.aggregate /path/to/local/results/<run_id>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _read_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def _trajectory_summary(traj_dir: Path) -> dict | None:
    manifest = _read_json(traj_dir / "manifest.json")
    if not manifest:
        return None
    trajectory = _read_jsonl(traj_dir / "trajectory.jsonl")
    state = _read_json(traj_dir / "state.json")

    # Best metric from AIDE's journal (use it as the trajectory's outcome).
    journal_match = list(traj_dir.rglob("journal.json"))
    journal = json.loads(journal_match[0].read_text()) if journal_match else {}
    metrics = [
        n["metric"]
        for n in journal.get("nodes", [])
        if n.get("metric") and isinstance(n["metric"].get("value"), (int, float))
    ]
    if metrics:
        # Respect maximize flag from any node (they should be uniform).
        maximize = metrics[0].get("maximize", True)
        chooser = max if maximize else min
        best = chooser(metrics, key=lambda m: m["value"])["value"]
    else:
        best = None

    total_in = sum((r["usage"]["input_tokens"] for r in trajectory), 0)
    total_out = sum((r["usage"]["output_tokens"] for r in trajectory), 0)
    error_nodes = sum(1 for r in trajectory if r.get("output", {}).get("errors"))

    stage_counts: Counter[str] = Counter()
    for r in trajectory:
        stage_counts[r["stage"]["sub_stage"]] += 1

    return {
        "task": manifest.get("task", {}).get("name"),
        "cell": "with_skill" if manifest.get("cell", {}).get("with_skill") else "without_skill",
        "seed": manifest.get("seed"),
        "status": manifest.get("result", {}).get("status"),
        "wall_clock_sec": manifest.get("timestamps", {}).get("wall_clock_sec"),
        "best_metric": best,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "error_nodes": error_nodes,
        "stage_counts": dict(stage_counts),
        "predicates_passed": sum(1 for v in state.values() if v is True),
        "predicates_total": sum(1 for v in state.values() if isinstance(v, bool)),
    }


def aggregate(run_dir: Path) -> dict:
    summaries: list[dict] = []
    for traj in sorted(run_dir.iterdir()):
        if not traj.is_dir():
            continue
        s = _trajectory_summary(traj)
        if s:
            s["trajectory_id"] = traj.name
            summaries.append(s)

    # Paired Lift per (task, seed): with - without.
    paired: dict[tuple[str, int], dict[str, float | None]] = defaultdict(dict)
    for s in summaries:
        key = (s["task"], s["seed"])
        paired[key][s["cell"]] = s["best_metric"]
    lifts: list[float] = []
    for (_task, _seed), cells in paired.items():
        w = cells.get("with_skill")
        wo = cells.get("without_skill")
        if w is not None and wo is not None:
            lifts.append(w - wo)
    lift_mean = statistics.mean(lifts) if lifts else None
    lift_stdev = statistics.stdev(lifts) if len(lifts) >= 2 else None
    n_pairs = len(lifts)
    ci95 = 1.96 * lift_stdev / (n_pairs**0.5) if lift_stdev and n_pairs else None

    return {
        "trajectory_count": len(summaries),
        "paired_lifts": lifts,
        "lift_mean": lift_mean,
        "lift_stdev": lift_stdev,
        "lift_ci95_halfwidth": ci95,
        "pair_count": n_pairs,
        "trajectories": summaries,
    }


def write_markdown(report: dict, out: Path) -> None:
    lines: list[str] = [
        "# A/B sweep summary",
        "",
        f"- trajectories: **{report['trajectory_count']}**",
        f"- paired (task,seed) cells: **{report['pair_count']}**",
        f"- mean Lift (with − without): **{report['lift_mean']}**",
        f"- Lift 95% CI half-width: **{report['lift_ci95_halfwidth']}**",
        "",
        "## Per-trajectory",
        "",
        "| task | cell | seed | status | best_metric | tok_in | tok_out | err | preds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report["trajectories"]:
        preds = f"{s['predicates_passed']}/{s['predicates_total']}"
        lines.append(
            f"| {s['task']} | {s['cell']} | {s['seed']} | {s['status']} | "
            f"{s['best_metric']} | {s['input_tokens']} | {s['output_tokens']} | "
            f"{s['error_nodes']} | {preds} |"
        )
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate trajectory outputs")
    parser.add_argument("run_dir", type=Path, help="Directory containing per-trajectory subdirs")
    args = parser.parse_args(argv)
    report = aggregate(args.run_dir)
    (args.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, args.run_dir / "report.md")
    print(f"[aggregate] wrote {args.run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
