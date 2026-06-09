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
from . import stage_metrics as _stage_metrics
from . import state_predicates as _state_predicates

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
    # Prefer the entrypoint-written state.json; if absent (the MLEvolve
    # entrypoint did not run state_predicates), compute the generic predicates
    # inline from the journal/artifacts so predicate pass-rates aren't blank.
    state = _read_json(traj_dir / "state.json")
    if not state:
        try:
            state = _state_predicates.evaluate(traj_dir)
        except Exception:
            state = {}

    # Self-reported best metric from the search journal (the agent's own
    # stdout number — gameable; kept only for drift diagnostics). Reuse the
    # defensive picker: a trajectory may hold several journals from retried
    # attempts, some empty/truncated.
    journal = _metrics._find_journal(traj_dir) or {}
    metrics = [
        n["metric"]
        for n in journal.get("nodes", [])
        if n.get("metric") and isinstance(n["metric"].get("value"), (int, float))
    ]
    if metrics:
        # Respect maximize flag from any node (they should be uniform). Note
        # .get("maximize", True) is NOT enough: the key can be present-but-None
        # in the serialized journal, which would select min. Coerce safely.
        _mx = metrics[0].get("maximize")
        maximize = _mx if isinstance(_mx, bool) else True
        chooser = max if maximize else min
        self_reported = chooser(metrics, key=lambda m: m["value"])["value"]
    else:
        self_reported = None

    # Trustworthy outcome: the held-out grader's score (mleval.grader scored
    # the best node's preserved predictions against held-out references).
    #   - grader ran + valid     → best_metric = held-out score
    #   - grader ran + invalid   → best_metric = None (drift correctly scores 0/none)
    #   - grader never ran (legacy run, no held_out_score.json) → fall back to
    #     the self-reported number so old aggregations don't all go blank.
    held = _read_json(traj_dir / "held_out_score.json")
    if held:
        held_valid = bool(held.get("valid")) and isinstance(held.get("score"), (int, float))
        best = held["score"] if held_valid else None
    else:
        held_valid = None  # grader did not run
        best = self_reported

    # trajectory.jsonl is the adapter's schema-versioned record (schema 1.0):
    # per-node token totals are flat fields, "buggy" replaces the old
    # output.errors list, and "stage" is the flat MCGS stage string.
    total_in = sum((r.get("llm_total_in_tokens") or 0) for r in trajectory)
    total_out = sum((r.get("llm_total_out_tokens") or 0) for r in trajectory)
    error_nodes = sum(1 for r in trajectory if r.get("is_buggy"))

    # Two granularities of stage activity:
    #   stage_counts     — the 5 MCGS stages (root/draft/debug/improve/evolution)
    #   sub_stage_counts — the 16 pipeline sub-stages from the multi-label
    #                      classifier (stage_classifier.all_sub_stages). This is
    #                      the L2a granularity the Stage-2 plan reports on, and
    #                      what stage_chi_square tests by default.
    stage_counts: Counter[str] = Counter()
    sub_stage_counts: Counter[str] = Counter()
    for r in trajectory:
        stage = r.get("stage")
        if stage:
            stage_counts[stage] += 1
        for sub in (r.get("stage_classifier") or {}).get("all_sub_stages") or []:
            sub_stage_counts[sub] += 1

    try:
        per_sub_stage = _stage_metrics.stage_metrics(traj_dir)
    except Exception:
        per_sub_stage = {}

    derived = _metrics.per_trajectory(traj_dir, REPO_ROOT)
    return {
        "task": manifest.get("task", {}).get("name"),
        "cell": manifest.get("cell", {}).get("name", "without_skill"),
        "seed": manifest.get("seed"),
        "status": manifest.get("result", {}).get("status"),
        "wall_clock_sec": manifest.get("timestamps", {}).get("wall_clock_sec"),
        "best_metric": best,
        # Drift diagnostics: a large gap between self_reported and a None/low
        # held-out score is the signature of an off-task trajectory.
        "self_reported_metric": self_reported,
        "held_out_valid": held_valid,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "error_nodes": error_nodes,
        "stage_counts": dict(stage_counts),
        "sub_stage_counts": dict(sub_stage_counts),
        "stage_metrics": per_sub_stage,
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
        f"p≈{_fmt(chi.get('p_value_approx'))}  n_stages={_fmt(chi.get('n_stages'))}  "
        f"(over `{chi.get('counts_key', 'sub_stage_counts')}`)",
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

    _write_substage_section(report, lines)

    out.write_text("\n".join(lines) + "\n")


def _write_substage_section(report: dict, lines: list[str]) -> None:
    """L2a per-sub-stage decomposition: clean-reach · rework · failure-modes.

    Uses the multi-label classifier's ``all_sub_stages``; reachability/rework/
    failure-modes attribute cleanly even though MLEvolve's monolithic
    full-pipeline nodes touch many sub-stages at once (the co-location
    confound that blocks per-stage time/token attribution). When exactly one
    with_skill and one without_skill trajectory are present, emit a paired
    A/B table; otherwise one block per trajectory.
    """
    sm = _stage_metrics
    trajs = report["trajectories"]
    withs = [s for s in trajs if s["cell"] == "with_skill" and s.get("stage_metrics")]
    withouts = [s for s in trajs if s["cell"] == "without_skill" and s.get("stage_metrics")]

    lines += ["", "## L2a per-sub-stage — clean-reach · rework · failure-modes", ""]

    if len(withs) == 1 and len(withouts) == 1:
        mw, mn = withs[0]["stage_metrics"], withouts[0]["stage_metrics"]
        subs = sm._all_substages(mw, mn)
        lines += [
            f"Paired: with_skill s{withs[0]['seed']} vs without_skill s{withouts[0]['seed']}.",
            "", "### clean-reach (clean nodes / nodes touching stage)", "",
            "| Sub-stage | Label | with_skill | without_skill |", "|---|---|---|---|",
        ]
        for s in subs:
            lines.append(f"| {s} | {sm._LABELS.get(s, '?')} | "
                         f"{sm._fmt_reach(mw.get(s))} | {sm._fmt_reach(mn.get(s))} |")
        lines += ["", "### rework (re-attempts beyond first = touches − 1)", "",
                  "| Sub-stage | Label | with_skill | without_skill |", "|---|---|---|---|"]
        for s in subs:
            rw_w = mw.get(s, {}).get("rework", "—")
            rw_n = mn.get(s, {}).get("rework", "—")
            lines.append(f"| {s} | {sm._LABELS.get(s, '?')} | {rw_w} | {rw_n} |")
        lines += ["", "### failure modes (exc_type over buggy nodes touching stage)", "",
                  "| Sub-stage | Label | with_skill | without_skill |", "|---|---|---|---|"]
        for s in subs:
            lines.append(f"| {s} | {sm._LABELS.get(s, '?')} | "
                         f"{sm._fmt_modes(mw.get(s))} | {sm._fmt_modes(mn.get(s))} |")
    else:
        for s in trajs:
            m = s.get("stage_metrics") or {}
            if not m:
                continue
            lines += [f"### {s['cell']} s{s['seed']}", "",
                      "| Sub-stage | Label | clean-reach | rework | failure modes |",
                      "|---|---|---|---|---|"]
            for sub in sm._all_substages(m):
                a = m.get(sub)
                lines.append(f"| {sub} | {sm._LABELS.get(sub, '?')} | {sm._fmt_reach(a)} "
                             f"| {a['rework'] if a else '—'} | {sm._fmt_modes(a)} |")
            lines.append("")


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
