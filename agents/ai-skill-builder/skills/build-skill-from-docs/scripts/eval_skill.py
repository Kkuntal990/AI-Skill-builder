#!/usr/bin/env python3
"""Skill evaluation harness (CLI) — measures triggering F1, functional pass rate
(with-skill vs without-skill A/B), organic activation, citation rate, and token
cost on an installed OpenClaw skill.

Implements the methodology from Anthropic's `skill-creator` (anthropics/skills).

As of the Skills-phase3 "tester-owns-eval" refactor, the whole behavioral-eval
stack is self-contained in `eval_core.py` (triggering judge, decoy list, sibling
loader, description optimizer all moved there). This file — and `eval_core` — live
inside the **skill-tester** agent's `evaluate-skill` skill; the `ai-skill-builder`
agent no longer imports either, it delegates gating to skill-tester via a sub-agent
call. This module imports NOTHING from `skill_builder`.

Usage:
    python3 eval_skill.py triggering    <skill-dir>
    python3 eval_skill.py functional    <skill-dir> [--runs N]
    python3 eval_skill.py activation    <skill-dir> [--runs N] [--model opus]
    python3 eval_skill.py all           <skill-dir> [--runs N] [--with-activation]
    python3 eval_skill.py gate          <skill-dir> [--profile smoke|full]   # verdict JSON
    python3 eval_skill.py baseline-probe <skill-dir?> --intent <brief>        # gap_notes JSON
    python3 eval_skill.py report        <skill-dir>
    python3 eval_skill.py pass-bar      <skill-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import eval_core as ec  # noqa: E402


def _decoys_for(skill_dir: Path, args: argparse.Namespace) -> tuple[list[dict], str]:
    """Real co-resident siblings by default (--siblings <dir> or the install root);
    fall back to canned DECOY_SKILLS only when <2 siblings exist."""
    meta = ec.load_skill_meta(skill_dir)
    sib_dir = getattr(args, "siblings", None) or str(skill_dir.parent)
    sibs = ec.load_sibling_descriptions(sib_dir, exclude_name=meta["name"])
    return (sibs, "siblings") if len(sibs) >= 2 else (ec.DECOY_SKILLS, "decoys")


def cmd_triggering(args: argparse.Namespace) -> dict:
    """Triggering F1: judge each prompt over the target skill + competitors (P1.1)."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    decoys, label = _decoys_for(skill_dir, args)
    return ec.run_triggering(skill_dir, ec.judge_triggering, decoys,
                             runs=args.runs, competitors_label=label)


def cmd_functional(args: argparse.Namespace) -> dict:
    """Functional with/without-skill A/B, deterministic assertion scoring (+ optional
    LLM grader for test['judge'] assertions via --llm-grader)."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    grader = ec.make_claude_grader() if getattr(args, "llm_grader", False) else None
    return ec.run_functional(skill_dir, agent=args.agent, runs=args.runs,
                             per_prompt_timeout=args.per_prompt_timeout, grader_fn=grader)


def cmd_activation(args: argparse.Namespace) -> dict:
    """Organic-activation eval on `claude -p` (Skills-3.0 Phase 3-3)."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    return ec.run_activation(skill_dir, model=getattr(args, "model", "") or "",
                             runs=args.runs, per_prompt_timeout=args.per_prompt_timeout,
                             max_concurrency=getattr(args, "max_concurrency", 3))


def cmd_optimize_description(args: argparse.Namespace) -> dict:
    """P1.3: optimize the skill's description on a 60/40 held-out split of triggering.json."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    decoys, _ = _decoys_for(skill_dir, args)
    return ec.optimize_description(skill_dir, ec.judge_triggering, decoys, ec.improve_description,
                                   holdout=args.holdout, runs=args.runs, max_iters=args.max_iters)


def cmd_gate(args: argparse.Namespace) -> dict:
    """Behavioral ship-gate verdict (what the builder delegates to skill-tester).
    Reads evals/triggering.json in the dir, runs the profile's eval suite, scores
    vs pass_bar.json, and emits the verdict JSON the builder consumes."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    return ec.run_gate(skill_dir, profile=getattr(args, "profile", "smoke"),
                       siblings_dir=getattr(args, "siblings", "") or "")


def cmd_baseline_probe(args: argparse.Namespace) -> dict:
    """P0.3 baseline-first probe: run the base agent (no skill) on intent-derived
    questions, return {gap_notes}. Intent+docs come from --payload-file (JSON
    {intent, doc_excerpt, n_questions}, how the builder delegates) or from --intent
    [+ --doc-file]."""
    intent = getattr(args, "intent", "") or ""
    doc_text = ""
    n_q = getattr(args, "n_questions", 2)
    pf = getattr(args, "payload_file", "") or ""
    if pf:
        try:
            p = json.loads(Path(pf).expanduser().read_text())
            intent = p.get("intent", intent) or intent
            doc_text = p.get("doc_excerpt", "") or ""
            n_q = int(p.get("n_questions", n_q) or n_q)
        except (OSError, ValueError, TypeError):
            pass
    else:
        doc_file = getattr(args, "doc_file", "") or ""
        if doc_file:
            try:
                doc_text = Path(doc_file).expanduser().read_text(errors="replace")
            except OSError:
                doc_text = ""
    if not intent:
        return {"gap_notes": "", "error": "no intent (pass --intent or --payload-file)"}
    return ec.baseline_probe(intent, doc_text, agent=getattr(args, "agent", "main") or "main",
                             n_questions=n_q, timeout=getattr(args, "per_prompt_timeout", 180))


def cmd_all(args: argparse.Namespace) -> dict:
    triggering = cmd_triggering(args)
    functional = cmd_functional(args)
    out = {"triggering": triggering, "functional": functional}
    if getattr(args, "with_activation", False):
        out["activation"] = cmd_activation(args)
    return out


def _load_latest_results(skill_dir: Path) -> tuple[dict | None, dict | None, dict | None]:
    """Read the most recent triggering / functional / activation results off disk."""
    triggering_path = ec.find_latest(skill_dir, "triggering") or ec.find_latest(skill_dir, "all")
    functional_path = ec.find_latest(skill_dir, "functional") or ec.find_latest(skill_dir, "all")
    activation_path = ec.find_latest(skill_dir, "activation") or ec.find_latest(skill_dir, "all")
    triggering = functional = activation = None
    if triggering_path:
        d = json.loads(triggering_path.read_text())
        triggering = d.get("triggering", d) if "triggering" in d else d
    if functional_path:
        d = json.loads(functional_path.read_text())
        functional = d.get("functional", d) if "functional" in d else d
    if activation_path:
        d = json.loads(activation_path.read_text())
        activation = d.get("activation") if "activation" in d else (
            d if "should_trigger" in d and d.get("executor") else None)
    return triggering, functional, activation


def cmd_report(args: argparse.Namespace) -> dict:
    """Read latest grading_results/*.json + render markdown summary."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    triggering, functional, activation = _load_latest_results(skill_dir)
    pass_bar = ec.load_pass_bar(skill_dir)
    md, _ = ec.build_report(skill_dir, triggering, functional, pass_bar, activation)
    return {"markdown": md}


def cmd_pass_bar(args: argparse.Namespace) -> dict:
    """Re-load latest results, evaluate against pass_bar.json, set return code."""
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    triggering, functional, activation = _load_latest_results(skill_dir)
    pass_bar = ec.load_pass_bar(skill_dir)
    _, overall = ec.build_report(skill_dir, triggering, functional, pass_bar, activation)
    args._exit_code = 0 if overall else 1  # picked up in main()
    return {"pass": overall, "pass_bar": pass_bar}


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval_skill.py", description="Skill evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = lambda sp: (
        sp.add_argument("skill_dir"),
        sp.add_argument("--runs", type=int, default=3, help="Triggering runs per prompt (default 3)"),
        sp.add_argument("--agent", default=ec.FUNCTIONAL_EXECUTOR,
                        help="OpenClaw agent for functional runs (default skill-eval-target)"),
        sp.add_argument("--per-prompt-timeout", type=int, default=240),
        sp.add_argument("--siblings", default="",
                        help="dir of co-resident skills to judge triggering against "
                             "(default: the target skill's parent dir; falls back to canned decoys if <2)"),
        sp.add_argument("--llm-grader", action="store_true",
                        help="grade test['judge'] assertions with a claude -p grader (P1.5)"),
    )

    t = sub.add_parser("triggering", help="Triggering F1 only")
    common(t); t.set_defaults(func=cmd_triggering)

    f = sub.add_parser("functional", help="Functional A/B only")
    common(f); f.set_defaults(func=cmd_functional)

    ac = sub.add_parser("activation",
                        help="Organic-activation eval on `claude -p` (Skills-3.0 Phase 3-3)")
    common(ac)
    ac.add_argument("--model", default="",
                    help=f"claude -p model (default ${{SKILLBUILD_LLM_MODEL}} = {ec.SKILLBUILD_LLM_MODEL})")
    ac.add_argument("--max-concurrency", type=int, default=3,
                    help="parallel `claude -p` executors (default 3)")
    ac.set_defaults(func=cmd_activation)

    a = sub.add_parser("all", help="Triggering + functional [+ activation]")
    common(a)
    a.add_argument("--model", default="", help="claude -p model for --with-activation")
    a.add_argument("--max-concurrency", type=int, default=3)
    a.add_argument("--with-activation", action="store_true",
                   help="also run the organic-activation executor (needs the `claude` CLI)")
    a.set_defaults(func=cmd_all)

    od = sub.add_parser("optimize-description",
                        help="P1.3: held-out (60/40) description optimization")
    od.add_argument("skill_dir")
    od.add_argument("--runs", type=int, default=3)
    od.add_argument("--siblings", default="")
    od.add_argument("--holdout", type=float, default=0.4)
    od.add_argument("--max-iters", type=int, default=5, dest="max_iters")
    od.set_defaults(func=cmd_optimize_description)

    g = sub.add_parser("gate", help="Behavioral ship-gate verdict JSON (builder delegates here)")
    g.add_argument("skill_dir")
    g.add_argument("--profile", choices=["smoke", "full"], default="smoke",
                   help="smoke=triggering runs=1; full adds organic activation")
    g.add_argument("--siblings", default="",
                   help="dir of co-resident skills to judge triggering against")
    g.set_defaults(func=cmd_gate)

    bp = sub.add_parser("baseline-probe",
                        help="P0.3: run base agent (no skill) on intent questions -> gap_notes JSON")
    bp.add_argument("--intent", default="", help="the skill intent brief (or use --payload-file)")
    bp.add_argument("--payload-file", default="", dest="payload_file",
                    help="JSON {intent, doc_excerpt, n_questions} — how the builder delegates")
    bp.add_argument("--doc-file", default="", help="optional path to source docs (excerpt used)")
    bp.add_argument("--agent", default="main", help="OpenClaw agent for baseline runs (never skill-tester)")
    bp.add_argument("--n-questions", type=int, default=2, dest="n_questions")
    bp.add_argument("--per-prompt-timeout", type=int, default=180)
    bp.set_defaults(func=cmd_baseline_probe)

    r = sub.add_parser("report", help="Render markdown summary of latest grading results")
    r.add_argument("skill_dir")
    r.set_defaults(func=cmd_report)

    pb = sub.add_parser("pass-bar", help="Evaluate pass_bar.json — exit code 0 PASS / 1 FAIL")
    pb.add_argument("skill_dir")
    pb.set_defaults(func=cmd_pass_bar)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.func(args)
    if args.cmd == "report":
        print(out["markdown"])
    else:
        print(json.dumps(out, indent=2, default=str))
    # Persist results / report. Skip pass-bar (just an exit code) + baseline-probe (no dir).
    try:
        if not getattr(args, "skill_dir", None):
            return getattr(args, "_exit_code", 0)
        skill_dir = Path(args.skill_dir).expanduser().resolve()
        out_dir = skill_dir / "evals" / "grading_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        if args.cmd == "report":
            (out_dir / f"report-{ts}.md").write_text(out["markdown"])
        elif args.cmd != "pass-bar":
            (out_dir / f"{args.cmd}-{ts}.json").write_text(json.dumps(out, indent=2, default=str))
    except OSError:
        pass
    return getattr(args, "_exit_code", 0)


if __name__ == "__main__":
    sys.exit(main())
