"""Derived metrics over a single trajectory directory.

All functions read the raw artifacts that ``backend_wrapper`` /
``interpreter_patch`` / ``adapter_aide`` already wrote to disk. Nothing here
runs during the agent — these are pure post-hoc derivations called from
``aggregate.py`` (locally, on Mac) after ``kubectl cp`` pulls a sweep.

Inputs available per trajectory dir:

    manifest.json            – run-level metadata (cell, task, llm_model)
    trajectory.jsonl         – one record per AIDE node (from adapter)
    state.json               – generic + per-task predicate booleans
    prompts.jsonl            – raw per-LLM-call records (req_time, tokens)
    code/op_NNN.py           – per-step generated code (from adapter)
    working_dirs/op_NNN/     – per-step working_dir snapshot (from interpreter_patch)
    aide_logs/**/journal.json – AIDE's native journal (per-node metric, parent edges)

Each function returns ``None`` rather than raising when its input is absent —
sweeps with partial trajectories must still aggregate.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from . import pricing


# ---- I/O helpers ---------------------------------------------------------


def _read_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return out


def _find_journal(traj_dir: Path) -> dict | None:
    matches = list(traj_dir.rglob("journal.json"))
    if not matches:
        return None
    return _read_json(matches[0])


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile, stdlib only."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ---- 1. Cost (USD) -------------------------------------------------------


def cost_usd(traj_dir: Path) -> float | None:
    manifest = _read_json(traj_dir / "manifest.json") or {}
    model = manifest.get("agent", {}).get("llm_model")
    if not model:
        return None
    prompts = _read_jsonl(traj_dir / "prompts.jsonl")
    in_tok = sum(p.get("in_tokens", 0) or 0 for p in prompts)
    out_tok = sum(p.get("out_tokens", 0) or 0 for p in prompts)
    return pricing.cost_usd(model, in_tok, out_tok)


# ---- 2. API/tool call count ---------------------------------------------


def llm_call_count(traj_dir: Path) -> int:
    return len(_read_jsonl(traj_dir / "prompts.jsonl"))


# ---- 3. LLM-call latency (p50/p95) --------------------------------------


def llm_latency(traj_dir: Path) -> dict[str, float | None]:
    prompts = _read_jsonl(traj_dir / "prompts.jsonl")
    times = [p["req_time_sec"] for p in prompts if isinstance(p.get("req_time_sec"), (int, float))]
    return {
        "p50": _percentile(times, 0.50),
        "p95": _percentile(times, 0.95),
        "max": max(times) if times else None,
    }


# ---- 4. Per-step execution time (p50/p95) -------------------------------


def step_exec_time(traj_dir: Path) -> dict[str, float | None]:
    """AIDE's interpreter exec_time per node (excludes LLM time)."""
    journal = _find_journal(traj_dir) or {}
    times = [float(n["exec_time"]) for n in journal.get("nodes", []) if isinstance(n.get("exec_time"), (int, float))]
    return {
        "p50": _percentile(times, 0.50),
        "p95": _percentile(times, 0.95),
        "total": sum(times) if times else None,
    }


# ---- 5. Step count (path length) ----------------------------------------


def step_count(traj_dir: Path) -> int:
    """Count AIDE journal nodes (one per code-gen/exec step)."""
    journal = _find_journal(traj_dir) or {}
    return len(journal.get("nodes", []))


# ---- 6. Redundant-loop count + 7. Self-correction success ---------------


def _walk_parent_edges(journal: dict) -> list[tuple[dict, dict | None]]:
    """Yield (child, parent_or_None) pairs in node order."""
    nodes = journal.get("nodes", [])
    node2parent = journal.get("node2parent", {})
    by_id = {n["id"]: n for n in nodes}
    pairs: list[tuple[dict, dict | None]] = []
    for n in sorted(nodes, key=lambda x: x.get("step", 0)):
        pid = node2parent.get(n["id"])
        pairs.append((n, by_id.get(pid) if pid else None))
    return pairs


def redundant_loops(traj_dir: Path) -> int | None:
    """Count debug attempts where parent was also buggy AND the fix failed.

    Defined as: (child.is_buggy ∧ parent.is_buggy). A high count means the
    agent kept trying to patch the same broken code without success.
    """
    journal = _find_journal(traj_dir)
    if not journal:
        return None
    count = 0
    for child, parent in _walk_parent_edges(journal):
        if parent is None:
            continue
        if child.get("is_buggy") and parent.get("is_buggy"):
            count += 1
    return count


def self_correction_rate(traj_dir: Path) -> float | None:
    """Fraction of debug attempts (parent.is_buggy) that produced a non-buggy child."""
    journal = _find_journal(traj_dir)
    if not journal:
        return None
    attempts = 0
    successes = 0
    for child, parent in _walk_parent_edges(journal):
        if parent is None or not parent.get("is_buggy"):
            continue
        attempts += 1
        if not child.get("is_buggy"):
            successes += 1
    if attempts == 0:
        return None
    return successes / attempts


# ---- 8. Hallucination rate ----------------------------------------------

_HALLUCINATION_PATTERNS = re.compile(
    r"\b(ImportError|ModuleNotFoundError|NameError|AttributeError)\b"
)


def hallucination_rate(traj_dir: Path) -> dict[str, float | int | None]:
    """Fraction of error records mentioning import/name/attribute errors.

    Proxy for the agent inventing modules, symbols, or attributes that don't
    exist. Computed over trajectory.jsonl records that have any error.
    """
    traj = _read_jsonl(traj_dir / "trajectory.jsonl")
    if not traj:
        return {"rate": None, "hallucinated": 0, "errored": 0}
    errored = 0
    hallucinated = 0
    for r in traj:
        errs = r.get("output", {}).get("errors") or []
        if not errs:
            continue
        errored += 1
        if any(_HALLUCINATION_PATTERNS.search(e) for e in errs):
            hallucinated += 1
    rate = hallucinated / errored if errored else None
    return {"rate": rate, "hallucinated": hallucinated, "errored": errored}


# ---- 11. Convergence curve ----------------------------------------------


def convergence_curve(traj_dir: Path) -> dict[str, Any] | None:
    """Per-step best_metric_so_far. Includes maximize flag for plot orientation."""
    journal = _find_journal(traj_dir)
    if not journal:
        return None
    nodes = sorted(journal.get("nodes", []), key=lambda n: n.get("step", 0))
    steps = []
    values = []
    best_so_far = []
    maximize_seen: set[bool] = set()
    running_best: float | None = None
    for n in nodes:
        m = n.get("metric") or {}
        v = m.get("value")
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        mx = m.get("maximize", True)
        maximize_seen.add(bool(mx))
        if running_best is None:
            running_best = v
        else:
            running_best = max(running_best, v) if mx else min(running_best, v)
        steps.append(n.get("step"))
        values.append(v)
        best_so_far.append(running_best)
    if not values:
        return None
    return {
        "steps": steps,
        "values": values,
        "best_so_far": best_so_far,
        "maximize": next(iter(maximize_seen)) if len(maximize_seen) == 1 else None,
    }


# ---- 12. Time to first valid submission ---------------------------------


def time_to_first_valid_submission(
    traj_dir: Path,
    required_columns: tuple[str, ...] = ("Id",),
) -> dict[str, int | None]:
    """Earliest op_NNN step whose snapshot has a parseable submission.csv.

    Looks under ``working_dirs/op_*/working/submission.csv`` and returns the
    step index of the first CSV whose header contains every column in
    ``required_columns``. Task-agnostic on purpose; per-task callers can
    override the column set.
    """
    snapshots = sorted((traj_dir / "working_dirs").glob("op_*"))
    for snap in snapshots:
        try:
            step = int(snap.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        for sub in snap.rglob("submission.csv"):
            try:
                with sub.open() as f:
                    header = next(csv.reader(f), None)
                if header and set(required_columns).issubset({c.strip() for c in header}):
                    return {"step": step, "submission_path": str(sub.relative_to(traj_dir))}
            except OSError:
                continue
    return {"step": None, "submission_path": None}


# ---- 13. Skill-API adoption rate ----------------------------------------


_PY_FENCE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)


def _extract_skill_apis(skill_md_path: Path) -> tuple[set[str], set[str]]:
    """Parse SKILL.md, AST-walk every python code fence, return (imports, calls).

    Anything inside ``` python … ``` (or ``` … ```) is treated as code. Bad
    blocks (unparseable) are silently skipped — SKILL.md often has pseudo-code
    or fragments. Returns empty sets if the file is absent.
    """
    if not skill_md_path.is_file():
        return set(), set()
    text = skill_md_path.read_text()
    imports: set[str] = set()
    calls: set[str] = set()
    for block in _PY_FENCE.findall(text):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
                for alias in node.names:
                    calls.add(alias.name)
            elif isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    calls.add(f.id)
                elif isinstance(f, ast.Attribute):
                    calls.add(f.attr)
    return imports, calls


def _extract_code_apis(code_text: str) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return set(), set()
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
            for alias in node.names:
                calls.add(alias.name)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                calls.add(f.id)
            elif isinstance(f, ast.Attribute):
                calls.add(f.attr)
    return imports, calls


def skill_api_adoption(traj_dir: Path, repo_root: Path) -> dict[str, Any] | None:
    """Fraction of code-gen steps that used at least one skill-recommended API.

    Resolves the skill via ``manifest.cell.skill_path`` → repo-local mirror at
    ``infra/skills/<basename>/SKILL.md``. Returns None for without_skill cells
    or when no skill_path is declared. Counts a step as "adopting" if its
    imports ∪ calls intersect the skill's imports ∪ calls.
    """
    manifest = _read_json(traj_dir / "manifest.json") or {}
    cell = manifest.get("cell", {}) or {}
    if cell.get("name") != "with_skill":
        return None
    skill_path = cell.get("skill_path") or ""
    if not skill_path:
        return None
    # In-pod path → repo-local mirror by basename of the parent dir.
    skill_dir_name = Path(skill_path).parent.name
    local_skill_md = repo_root / "infra" / "skills" / skill_dir_name / "SKILL.md"
    skill_imports, skill_calls = _extract_skill_apis(local_skill_md)
    if not skill_imports and not skill_calls:
        return {
            "adoption_rate": None,
            "steps_with_code": 0,
            "steps_adopting": 0,
            "skill_md_resolved": str(local_skill_md) if local_skill_md.is_file() else None,
            "reason": "skill_md_missing_or_no_code_blocks",
        }
    code_files = sorted((traj_dir / "code").glob("op_*.py"))
    steps_with_code = 0
    steps_adopting = 0
    for cf in code_files:
        try:
            code_text = cf.read_text()
        except OSError:
            continue
        if not code_text.strip():
            continue
        steps_with_code += 1
        imps, calls = _extract_code_apis(code_text)
        if (imps & skill_imports) or (calls & (skill_imports | skill_calls)):
            steps_adopting += 1
    rate = steps_adopting / steps_with_code if steps_with_code else None
    return {
        "adoption_rate": rate,
        "steps_with_code": steps_with_code,
        "steps_adopting": steps_adopting,
        "skill_md_resolved": str(local_skill_md),
        "skill_api_count": len(skill_imports) + len(skill_calls),
    }


# ---- Sweep-level (need both cells) --------------------------------------


def cost_normalized_lift(
    lift_mean: float | None,
    summaries: list[dict],
) -> dict[str, float | None]:
    """Lift per 1k tokens and per dollar, across all paired trajectories.

    Normalizes by the with_skill cell's compute spend (the cost of the help,
    not the baseline). Returns None entries when inputs are missing.
    """
    if lift_mean is None:
        return {"lift_per_1k_tokens": None, "lift_per_usd": None}
    with_skill = [s for s in summaries if s.get("cell") == "with_skill"]
    if not with_skill:
        return {"lift_per_1k_tokens": None, "lift_per_usd": None}
    tot_tokens = sum((s.get("input_tokens", 0) + s.get("output_tokens", 0)) for s in with_skill)
    tot_cost = sum((s.get("cost_usd") or 0.0) for s in with_skill)
    return {
        "lift_per_1k_tokens": (lift_mean / (tot_tokens / 1000)) if tot_tokens else None,
        "lift_per_usd": (lift_mean / tot_cost) if tot_cost else None,
    }


def stage_chi_square(summaries: list[dict]) -> dict[str, float | int | None]:
    """χ² test of stage_counts distribution between with_skill and without_skill.

    Stdlib-only, no scipy. Returns chi2 statistic, dof, and a rough p-value
    approximation via the survival function of the chi-square distribution.
    Cells with both observed=0 and expected=0 are dropped to avoid divide-by-0.
    """
    by_cell: dict[str, Counter[str]] = {"with_skill": Counter(), "without_skill": Counter()}
    for s in summaries:
        cell = s.get("cell")
        if cell in by_cell:
            for stage, n in (s.get("stage_counts") or {}).items():
                by_cell[cell][stage] += n
    stages = sorted(set(by_cell["with_skill"]) | set(by_cell["without_skill"]))
    if not stages:
        return {"chi2": None, "dof": None, "p_value_approx": None, "n_stages": 0}
    row_totals = {c: sum(by_cell[c].values()) for c in by_cell}
    grand = sum(row_totals.values())
    if grand == 0:
        return {"chi2": None, "dof": None, "p_value_approx": None, "n_stages": len(stages)}
    chi2 = 0.0
    dropped = 0
    for c in by_cell:
        for stage in stages:
            o = by_cell[c][stage]
            col_total = sum(by_cell[cc][stage] for cc in by_cell)
            e = row_totals[c] * col_total / grand
            if e == 0:
                dropped += 1
                continue
            chi2 += (o - e) ** 2 / e
    dof = (len(stages) - 1) * (len(by_cell) - 1)
    return {
        "chi2": chi2,
        "dof": dof,
        "p_value_approx": _chi2_sf(chi2, dof) if dof > 0 else None,
        "n_stages": len(stages),
        "cells_dropped_zero_expected": dropped,
    }


def _chi2_sf(x: float, k: int) -> float:
    """Survival function P(X > x) for χ²_k. Stdlib only; uses incomplete gamma.

    Accurate enough for reporting (≤ 1e-6 abs error for typical eval cases).
    """
    if x <= 0:
        return 1.0
    # P(X > x) = 1 - regularized_lower_incomplete_gamma(k/2, x/2)
    return 1.0 - _regularized_gamma_p(k / 2, x / 2)


def _regularized_gamma_p(a: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Series expansion for x < a+1, continued fraction otherwise. Standard."""
    if x == 0:
        return 0.0
    if x < a + 1:
        # Series: sum_{n=0..} x^n / Π_{i=0..n} (a+i)
        term = 1.0 / a
        total = term
        for n in range(1, max_iter):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * eps:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Lentz's continued fraction for Q(a, x) = 1 - P(a, x)
    b = x + 1 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, max_iter):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return 1.0 - h * math.exp(-x + a * math.log(x) - math.lgamma(a))


# ---- top-level: compute all per-trajectory metrics ----------------------


def per_trajectory(traj_dir: Path, repo_root: Path) -> dict[str, Any]:
    """Return all per-trajectory derived metrics as a flat dict."""
    return {
        "cost_usd": cost_usd(traj_dir),
        "llm_call_count": llm_call_count(traj_dir),
        "llm_latency": llm_latency(traj_dir),
        "step_exec_time": step_exec_time(traj_dir),
        "step_count": step_count(traj_dir),
        "redundant_loops": redundant_loops(traj_dir),
        "self_correction_rate": self_correction_rate(traj_dir),
        "hallucination": hallucination_rate(traj_dir),
        "convergence": convergence_curve(traj_dir),
        "first_valid_submission": time_to_first_valid_submission(traj_dir),
        "skill_api_adoption": skill_api_adoption(traj_dir, repo_root),
    }
