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

from . import metrics as _metrics

REPO_ROOT = Path(__file__).resolve().parents[3]


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

    derived = _metrics.per_trajectory(traj_dir, REPO_ROOT)
    return {
        "task": manifest.get("task", {}).get("name"),
        "cell": manifest.get("cell", {}).get("name", "without_skill"),
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
        # Derived metrics (see mleval.analyzer.metrics for definitions)
        "cost_usd": derived["cost_usd"],
        "llm_call_count": derived["llm_call_count"],
        "llm_latency": derived["llm_latency"],
        "step_exec_time": derived["step_exec_time"],
        "step_count": derived["step_count"],
        "redundant_loops": derived["redundant_loops"],
        "self_correction_rate": derived["self_correction_rate"],
        "hallucination": derived["hallucination"],
        "convergence": derived["convergence"],
        "first_valid_submission": derived["first_valid_submission"],
        "skill_api_adoption": derived["skill_api_adoption"],
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

    cost_norm = _metrics.cost_normalized_lift(lift_mean, summaries)
    chi2 = _metrics.stage_chi_square(summaries)

    return {
        "trajectory_count": len(summaries),
        "paired_lifts": lifts,
        "lift_mean": lift_mean,
        "lift_stdev": lift_stdev,
        "lift_ci95_halfwidth": ci95,
        "pair_count": n_pairs,
        "cost_normalized_lift": cost_norm,
        "stage_chi_square": chi2,
        "trajectories": summaries,
    }


def _fmt(v, spec: str = ".4g") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def write_markdown(report: dict, out: Path) -> None:
    cn = report.get("cost_normalized_lift") or {}
    chi = report.get("stage_chi_square") or {}
    lines: list[str] = [
        "# A/B sweep summary",
        "",
        "## L1 outcome",
        f"- trajectories: **{report['trajectory_count']}**",
        f"- paired (task, seed) cells: **{report['pair_count']}**",
        f"- mean Lift (with − without): **{_fmt(report['lift_mean'])}**",
        f"- Lift 95% CI half-width: **{_fmt(report['lift_ci95_halfwidth'])}**",
        f"- Lift per 1k with-skill tokens: **{_fmt(cn.get('lift_per_1k_tokens'), '.4g')}**",
        f"- Lift per with-skill $: **{_fmt(cn.get('lift_per_usd'), '.4g')}**",
        "",
        "## L2a stage-distribution shift",
        f"- χ²={_fmt(chi.get('chi2'))}  dof={_fmt(chi.get('dof'))}  "
        f"p≈{_fmt(chi.get('p_value_approx'))}  n_stages={_fmt(chi.get('n_stages'))}",
        "",
        "## Per-trajectory — outcome + cost",
        "",
        "| cell | seed | status | best_metric | tok_in | tok_out | cost_$ | calls | wall_s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report["trajectories"]:
        lines.append(
            f"| {s['cell']} | {s['seed']} | {s['status']} | "
            f"{_fmt(s['best_metric'])} | {s['input_tokens']} | {s['output_tokens']} | "
            f"{_fmt(s.get('cost_usd'))} | {_fmt(s.get('llm_call_count'))} | "
            f"{_fmt(s.get('wall_clock_sec'))} |"
        )

    lines += [
        "",
        "## Per-trajectory — process quality",
        "",
        "| cell | seed | steps | err | hallu | redund | self-fix | preds | adopt | first-sub@ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report["trajectories"]:
        hall = s.get("hallucination") or {}
        adopt = s.get("skill_api_adoption") or {}
        fvs = s.get("first_valid_submission") or {}
        preds = f"{s['predicates_passed']}/{s['predicates_total']}"
        adopt_str = (
            f"{_fmt(adopt.get('adoption_rate'), '.2f')}"
            f" ({adopt.get('steps_adopting', '—')}/{adopt.get('steps_with_code', '—')})"
            if adopt and adopt.get("adoption_rate") is not None
            else "—"
        )
        hall_str = (
            f"{_fmt(hall.get('rate'), '.2f')} ({hall.get('hallucinated', 0)}/{hall.get('errored', 0)})"
            if hall.get("rate") is not None
            else f"0/{hall.get('errored', 0)}"
        )
        lines.append(
            f"| {s['cell']} | {s['seed']} | {_fmt(s.get('step_count'))} | "
            f"{s['error_nodes']} | {hall_str} | {_fmt(s.get('redundant_loops'))} | "
            f"{_fmt(s.get('self_correction_rate'), '.2f')} | {preds} | "
            f"{adopt_str} | {_fmt(fvs.get('step'))} |"
        )

    lines += [
        "",
        "## Per-trajectory — latency",
        "",
        "| cell | seed | LLM p50 | LLM p95 | LLM max | step-exec p50 | step-exec p95 | step-exec total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["trajectories"]:
        ll = s.get("llm_latency") or {}
        se = s.get("step_exec_time") or {}
        lines.append(
            f"| {s['cell']} | {s['seed']} | {_fmt(ll.get('p50'), '.2f')} | "
            f"{_fmt(ll.get('p95'), '.2f')} | {_fmt(ll.get('max'), '.2f')} | "
            f"{_fmt(se.get('p50'), '.2f')} | {_fmt(se.get('p95'), '.2f')} | "
            f"{_fmt(se.get('total'), '.1f')} |"
        )

    lines += [
        "",
        "## Convergence curves (per-cell best-metric-so-far)",
        "",
    ]
    for s in report["trajectories"]:
        conv = s.get("convergence") or {}
        if not conv or not conv.get("best_so_far"):
            continue
        pairs = ", ".join(
            f"({step}, {_fmt(v, '.5f')})"
            for step, v in zip(conv["steps"], conv["best_so_far"])
        )
        direction = "↑" if conv.get("maximize") else "↓"
        lines.append(f"- **{s['cell']} s{s['seed']}** ({direction} better): {pairs}")

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
