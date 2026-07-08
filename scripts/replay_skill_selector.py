#!/usr/bin/env python3
"""Offline replay + scoring of the MLEvolve skill selector — no GPU, no rebuild.

Reads the RAW ``select_skills`` selector asks already recorded on the PVC (from
``prompts.jsonl``) for one trajectory or a whole run, scores each against the
per-task gold labels (``infra/tasks/<task>/skill_gold.json``), and simulates the
per-node injection caps so you can see what WOULD be injected under a different
(max_skills, max_refs) policy — all before spending a single GPU-hour.

This is M2 of docs/eval/skill-retrieval-design.md: "a selector-accuracy table on
real recorded contexts, before any GPU spend." It answers, per task:
  - recall over the gold-relevant skills (did the selector find them?)
  - wrong-family picks (did it select a gold-irrelevant skill?)
  - decline rate (how often did it return nothing?) and whether declines were
    correct (gold had no relevant skill)
  - cap impact (how many nodes' asks would the 3/3 caps truncate?)

Sources, in priority order per trajectory:
  1. prompts.jsonl  — the RAW selector ask (pre-cap). Works on mvp-032 and every
     run since. This is canonical for replay because caps are re-simulated here.
  2. selection_events.jsonl (node_selection) — the POST-cap decision, used only
     as a fallback for runs whose prompts.jsonl lacks select_skills records.

Usage:
    python scripts/replay_skill_selector.py <run_or_trajectory_dir> \
        [--task NAME] [--max-per-node 3] [--max-refs-per-node 3] [--json]

<dir> may be a single trajectory dir (has manifest.json) or a run dir whose
immediate subdirs are trajectory dirs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE_RE = re.compile(r"^Stage:\s*(\S+)", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _read_jsonl(fp: Path) -> list[dict]:
    if not fp.is_file():
        return []
    out = []
    for line in fp.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _read_json(fp: Path) -> dict:
    return json.loads(fp.read_text()) if fp.is_file() else {}


def load_gold(task: str, tasks_dir: Path) -> dict | None:
    g = _read_json(tasks_dir / task / "skill_gold.json")
    return g or None


def load_ref_map(skills_dir: Path) -> dict[str, set[str]]:
    """{skill_name: {reference filenames}} for accurate ``__all__`` expansion."""
    out: dict[str, set[str]] = {}
    if not skills_dir.is_dir():
        return out
    for d in sorted(skills_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or not (d / "SKILL.md").is_file():
            continue
        refs = {p.name for p in (d / "references").glob("*.md")} if (d / "references").is_dir() else set()
        out[d.name] = refs
    return out


# --------------------------------------------------------------------------- #
# Selector-ask extraction
# --------------------------------------------------------------------------- #

def _parse_selector_output(raw) -> list[dict]:
    """Return the selector's ``selections`` list from a prompts.jsonl output field.

    The field is whatever llm.query returned for the func-call, serialized by
    prompt_logger — normally a JSON string of ``{"selections": [...]}``. Be
    liberal: accept a dict, a JSON string, or a bare list.
    """
    obj = raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(obj, dict):
        sels = obj.get("selections", [])
    elif isinstance(obj, list):
        sels = obj
    else:
        return []
    return [s for s in sels if isinstance(s, dict) and s.get("skill_name")]


def extract_asks(traj_dir: Path) -> tuple[list[dict], str]:
    """Return (asks, source). Each ask: {stage, skills, refs:{skill:[files]}}."""
    asks: list[dict] = []
    for rec in _read_jsonl(traj_dir / "prompts.jsonl"):
        if rec.get("func_spec_name") != "select_skills":
            continue
        sels = _parse_selector_output(rec.get("output"))
        um = rec.get("user_message") or ""
        m = _STAGE_RE.search(um) if isinstance(um, str) else None
        asks.append({
            "stage": m.group(1) if m else None,
            "skills": [s["skill_name"] for s in sels],
            "refs": {s["skill_name"]: (s.get("references") or []) for s in sels},
        })
    if asks:
        return asks, "prompts.jsonl"
    # Fallback: post-cap decisions from selection telemetry.
    for e in _read_jsonl(traj_dir / "selection_events.jsonl"):
        if e.get("event") != "node_selection":
            continue
        asks.append({
            "stage": e.get("stage"),
            "skills": e.get("selected_skills") or [],
            "refs": e.get("selected_references") or {},
        })
    return asks, "selection_events.jsonl"


# --------------------------------------------------------------------------- #
# Scoring + cap simulation
# --------------------------------------------------------------------------- #

def score_ask(skills: list[str], gold: dict) -> dict:
    sel = set(skills)
    relevant = set(gold.get("relevant_skills", []))
    acceptable = set(gold.get("acceptable_skills", []))
    irrelevant = set(gold.get("irrelevant_skills", []))
    hits = sel & relevant
    recall = len(hits) / len(relevant) if relevant else None
    ok = sel & (relevant | acceptable)
    if sel:
        precision = len(ok) / len(sel)
    else:
        precision = 1.0 if not relevant else 0.0  # correct decline vs missed
    return {
        "recall": recall,
        "precision": precision,
        "wrong_family": sorted(sel & irrelevant),
        "declined": not sel,
        "correct_decline": (not sel) and (not relevant),
    }


def simulate_caps(ask: dict, ref_map: dict[str, set[str]],
                  max_skills: int, max_refs: int) -> dict:
    """What the 3/3 (or given) caps would inject for this raw ask."""
    skills = ask["skills"]
    kept_skills = skills[:max_skills]
    refs_used = 0
    refs_truncated = False
    for sk in kept_skills:
        listed = ask["refs"].get(sk, [])
        expanded = sorted(ref_map.get(sk, set())) if listed == ["__all__"] else \
            [r for r in listed if r in ref_map.get(sk, set())]
        for _ in expanded:
            if refs_used >= max_refs:
                refs_truncated = True
                break
            refs_used += 1
    return {
        "skills_truncated": len(skills) > max_skills,
        "refs_truncated": refs_truncated,
        "injected_skills": len(kept_skills),
        "injected_refs": refs_used,
    }


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def trajectory_dirs(root: Path) -> list[Path]:
    if (root / "manifest.json").is_file() or (root / "prompts.jsonl").is_file():
        return [root]
    return [d for d in sorted(root.iterdir()) if d.is_dir()]


def replay(root: Path, tasks_dir: Path, skills_dir: Path,
           max_skills: int, max_refs: int, task_override: str | None) -> dict:
    ref_map = load_ref_map(skills_dir)
    per_task: dict[str, dict] = {}
    for traj in trajectory_dirs(root):
        manifest = _read_json(traj / "manifest.json")
        task = task_override or manifest.get("task", {}).get("name") or "unknown"
        cell = manifest.get("cell", {}).get("name", "unknown")
        asks, source = extract_asks(traj)
        if not asks:
            continue
        gold = load_gold(task, tasks_dir)
        bucket = per_task.setdefault(task, {
            "task": task, "gold": gold, "records": 0, "trajectories": [],
            "recalls": [], "precisions": [], "declines": 0, "correct_declines": 0,
            "wrong_family_records": 0, "skills_trunc": 0, "refs_trunc": 0,
            "stages": Counter(), "sources": Counter(),
        })
        bucket["trajectories"].append({"dir": traj.name, "cell": cell})
        bucket["sources"][source] += 1
        for ask in asks:
            bucket["records"] += 1
            bucket["stages"][ask["stage"] or "unknown"] += 1
            caps = simulate_caps(ask, ref_map, max_skills, max_refs)
            bucket["skills_trunc"] += int(caps["skills_truncated"])
            bucket["refs_trunc"] += int(caps["refs_truncated"])
            if gold:
                sc = score_ask(ask["skills"], gold)
                bucket["recalls"].append(sc["recall"])
                bucket["precisions"].append(sc["precision"])
                bucket["declines"] += int(sc["declined"])
                bucket["correct_declines"] += int(sc["correct_decline"])
                bucket["wrong_family_records"] += int(bool(sc["wrong_family"]))
            else:
                bucket["declines"] += int(not ask["skills"])
    # finalize
    for b in per_task.values():
        n = b["records"] or 1
        b["mean_recall"] = _mean(b["recalls"])
        b["mean_precision"] = _mean(b["precisions"])
        b["empty_rate"] = b["declines"] / n
        b["confidence"] = (b["gold"] or {}).get("confidence")
    return {
        "caps": {"max_skills": max_skills, "max_refs": max_refs},
        "tasks": per_task,
    }


def _fmt(v, spec=".2f"):
    return format(v, spec) if isinstance(v, float) else ("—" if v is None else str(v))


def print_report(report: dict) -> None:
    caps = report["caps"]
    print(f"\n=== Selector replay (caps: {caps['max_skills']} skills / "
          f"{caps['max_refs']} refs per node) ===\n")
    print("| task | conf | recs | recall | prec | empty | wrong-fam | "
          "skills-trunc | refs-trunc | src |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for task in sorted(report["tasks"]):
        b = report["tasks"][task]
        src = ",".join(b["sources"].keys())
        conf = b.get("confidence") or ("no-gold" if not b["gold"] else "—")
        print(f"| {task} | {conf} | {b['records']} | "
              f"{_fmt(b['mean_recall'])} | {_fmt(b['mean_precision'])} | "
              f"{_fmt(b['empty_rate'])} ({b['declines']}) | "
              f"{b['wrong_family_records']} | {b['skills_trunc']} | "
              f"{b['refs_trunc']} | {src} |")
    print("\nColumns: recall/prec are means over selector nodes with gold; "
          "empty = declined-selection rate (n); wrong-fam = nodes picking a "
          "gold-irrelevant skill; *-trunc = nodes the caps would truncate.")
    # Loud flags
    for task in sorted(report["tasks"]):
        b = report["tasks"][task]
        if b["gold"] and b["mean_recall"] is not None and b["mean_recall"] < 0.5:
            print(f"  ⚠ {task}: mean recall {_fmt(b['mean_recall'])} < 0.5 — "
                  f"selector is MISSING gold-relevant skills (routing failure).")
        if b["empty_rate"] >= 0.5 and (not b["gold"] or b["gold"].get("relevant_skills")):
            print(f"  ⚠ {task}: {_fmt(b['empty_rate'])} of nodes declined — "
                  f"possible silently-emptied treatment (spike-023 signature).")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", type=Path, help="Trajectory dir or run dir (pulled from PVC)")
    p.add_argument("--task", default=None, help="Override task name (else read from manifest.json)")
    p.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "infra/tasks")
    p.add_argument("--skills-dir", type=Path, default=REPO_ROOT / "infra/skills")
    p.add_argument("--max-per-node", type=int, default=3)
    p.add_argument("--max-refs-per-node", type=int, default=3)
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of the table")
    args = p.parse_args(argv)

    if not args.dir.exists():
        print(f"ERROR: {args.dir} does not exist", file=sys.stderr)
        return 1
    report = replay(args.dir, args.tasks_dir, args.skills_dir,
                    args.max_per_node, args.max_refs_per_node, args.task)
    if not report["tasks"]:
        print("No select_skills records found (without_skill cell, or no "
              "prompts.jsonl/selection_events.jsonl under the given dir).",
              file=sys.stderr)
        return 1
    if args.json:
        # Counters aren't JSON-serializable as-is; coerce.
        for b in report["tasks"].values():
            b["stages"] = dict(b["stages"])
            b["sources"] = dict(b["sources"])
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
