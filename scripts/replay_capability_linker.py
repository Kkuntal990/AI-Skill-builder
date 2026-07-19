#!/usr/bin/env python3
"""Offline replay of the capability linker over recorded MLEvolve node contexts.

W6 / E2 (HLD §12-E2, plan §E2′). This is the *capability* sibling of
``scripts/replay_skill_selector.py``: instead of scoring the legacy per-skill
selector, it reconstructs a ``NodeProfile`` from each recorded ``select_skills``
context and runs the DETERMINISTIC layers of ``capability_linker`` (hard filter →
budget cap → dependency closure) over a capability library, emitting per-stage and
aggregate metrics plus a legacy-vs-capability exposure comparison.

Everything in the default path is DETERMINISTIC and zero-GPU — NO LLM calls. The
semantic selector (stage B of the linker) is an LLM step, so the deterministic
replay STUBS it to "admit all survivors up to the budget". The capability exposure
it reports is therefore an **UPPER BOUND** on what would actually be injected, not
the final selected set. The real, tighter number needs the LLM pass — available
via the OPTIONAL ``--with-selector`` flag (off by default; see its caveats below).

What it computes (HLD §12-E2), over the reconstructed nodes:
  - per-stage node counts + candidate counts (survivors after the hard filter) +
    hard-filtered counts by machine reason code;
  - EXPOSURE: legacy-all-units (a legacy-selected package exposes ALL its units;
    a declined/empty legacy selection exposes 0) vs capability-upper-bound
    (survivors capped by ``budget.max_units`` + dependency closure);
  - hard-constraint VIOLATION rate: fraction of nodes where a rendered/admitted
    unit still violates a KNOWN hard constraint of its profile — ~0 by
    construction, computed as an assertion that validates the filter.

Usage:
    python scripts/replay_capability_linker.py <run_or_trajectory_dir> \
        [--library built|all] [--max-units 3] [--json] [--with-selector]

<dir> may be a single trajectory dir (has prompts.jsonl) or a run dir whose
immediate subdirs are trajectory dirs (e.g. pulled-results/spike-018). Trajectories
with no ``select_skills`` records (every without-skill cell) contribute 0 nodes.

Library composition (``--library``):
  built  — the PRIMARY library: the two E1 built skills (peft-tuning + vllm-inference),
           body from infra/skills/<skill>/SKILL.md, references from its references/*.md.
  all    — built + the 5 corpus E1 distractor manifests (peft, vllm, transformer-lens,
           ray-data, verl) compiled from their capabilities.json with EMPTY references.
           Used to widen the exposure/distractor picture only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import types
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SIDECAR = REPO_ROOT / "infra/agents/mlevolve/mlevolve_sidecar"
_E1 = REPO_ROOT / "docs/skill-builder/capability-mvp/e1"
_SKILLS = REPO_ROOT / "infra/skills"


# --------------------------------------------------------------------------- #
# Import capability_linker (and its sibling replay loaders) by file path.
#
# capability_linker._schema_version() does a relative import
# (``from .capability_schema import SCHEMA_VERSION``), so it must load under a stub
# ``mlevolve_sidecar`` package carrying a ``__path__`` — exactly as
# tests/test_capability_linker.py does. No MLEvolve stub is needed for the core.
# --------------------------------------------------------------------------- #

def _load_sidecar(sub: str):
    pkg = "mlevolve_sidecar"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(_SIDECAR)]
        sys.modules[pkg] = m
    full = f"{pkg}.{sub}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _SIDECAR / f"{sub}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cl = _load_sidecar("capability_linker")
_load_sidecar("capability_schema")  # pre-resolve the relative import target
# Reuse the sibling replay's tiny stdlib-only loaders (contract: reuse where sensible).
rss = _load_path("replay_skill_selector", REPO_ROOT / "scripts" / "replay_skill_selector.py")

NodeProfile = cl.NodeProfile
STAGE_MAP = cl.STAGE_MAP
LinkBudget = cl.LinkBudget
parse_hardware = cl.parse_hardware

_UNKNOWN_HW = {"gpu_count": None, "vram_gb": None, "gpu_name": None, "raw": ""}

# Machine reason codes the deterministic filter can emit (for stable report keys).
REASON_CODES = (
    "stage_excluded", "security_unknown", "cpu_only", "gpu_count", "vram_gb",
    "runtime_absent", "dep_missing", "dep_incompatible",
)

# YAML frontmatter stripper — mirrors skill_retriever._parse_frontmatter (body only).
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


def _strip_frontmatter(skill_md: str) -> str:
    m = _FRONTMATTER_RE.match(skill_md)
    return m.group(2) if m else skill_md


# --------------------------------------------------------------------------- #
# Capability library
# --------------------------------------------------------------------------- #

def load_capability_library(skill_dirs, manifest_paths) -> list[dict]:
    """Build the ``cap_skills`` list the linker consumes from E1 manifests.

    ``manifest_paths`` : iterable of ``capabilities.json`` paths (E1 built/corpus).
    ``skill_dirs``     : where to pull the SKILL.md body + references/*.md from.
        Accepts either
          - a dict ``{manifest_skill_name: infra/skills/<skill> dir}``, or
          - a list parallel to ``manifest_paths`` (a ``None`` entry = no dir).
        A manifest with no matching dir loads with ``body=""`` and
        ``references={}`` (the corpus-distractor case).

    Returns ``list`` of ``{name, body, references, capabilities}`` dicts — the
    retriever-shaped skill dict ``capability_linker._build_registry`` expects.
    """
    manifest_paths = list(manifest_paths)
    dirs_by_name = skill_dirs if isinstance(skill_dirs, dict) else None
    dirs_by_index = None if dirs_by_name is not None else list(skill_dirs or [])

    lib: list[dict] = []
    for i, mp in enumerate(manifest_paths):
        manifest = json.loads(Path(mp).read_text())
        name = manifest.get("skill_name") or ""
        caps = manifest.get("capabilities") or []
        if dirs_by_name is not None:
            skill_dir = dirs_by_name.get(name)
        else:
            skill_dir = dirs_by_index[i] if i < len(dirs_by_index) else None

        body = ""
        references: dict[str, str] = {}
        if skill_dir is not None:
            skill_dir = Path(skill_dir)
            md = skill_dir / "SKILL.md"
            if md.is_file():
                body = _strip_frontmatter(md.read_text())
            ref_dir = skill_dir / "references"
            if ref_dir.is_dir():
                references = {p.name: p.read_text() for p in sorted(ref_dir.glob("*.md"))}
        lib.append({"name": name, "body": body, "references": references, "capabilities": caps})
    return lib


# The PRIMARY library: the two E1 built skills (manifest paths + infra/skills dirs).
_BUILT = [
    (_E1 / "peft-tuning-built" / "capabilities.json", _SKILLS / "peft-tuning"),
    (_E1 / "vllm-inference-built" / "capabilities.json", _SKILLS / "vllm-inference"),
]
# The 5 corpus E1 distractor manifests (empty references; body-less).
_CORPUS = [
    _E1 / "peft" / "capabilities.json",
    _E1 / "vllm" / "capabilities.json",
    _E1 / "transformer-lens" / "capabilities.json",
    _E1 / "ray-data" / "capabilities.json",
    _E1 / "verl" / "capabilities.json",
]


def build_library(mode: str) -> list[dict]:
    """``built`` -> the two built skills; ``all`` -> built + 5 corpus distractors."""
    manifests = [p for p, _ in _BUILT]
    dirs = [d for _, d in _BUILT]
    if mode == "all":
        manifests = manifests + list(_CORPUS)
        dirs = dirs + [None] * len(_CORPUS)
    return load_capability_library(dirs, manifests)


# --------------------------------------------------------------------------- #
# NodeProfile reconstruction (contract §"Recorded-context format")
# --------------------------------------------------------------------------- #

_STAGE_RE = re.compile(r"^Stage:\s*(\S+)", re.MULTILINE)
_COMPUTE_RE = re.compile(r"^Compute:\s*(.*)$", re.MULTILINE)
# Header-line markers that introduce a multi-line section body (the value is on
# the NEXT line, per capability_linker._selector_user's ``f"{marker}:\n{body}"``).
_SECTION_MARKERS = [
    ("Task:", "task_text"),
    ("Error output (tail):", "parent_error"),
    ("Root-cause analysis:", "parent_analysis"),
    ("Current solution code (head):", "parent_code"),
]


def _runtime_caps(hw: dict) -> set:
    """Conservative KNOWN-present runtime caps. python is always present; gpu /
    multi-gpu are added ONLY when hardware is known (HLD §10.3 three-valued)."""
    caps = {"python"}
    gc = hw.get("gpu_count")
    if hw.get("gpu_name") or (isinstance(gc, int) and gc > 0):
        caps.add("gpu")
    if isinstance(gc, int) and gc > 1:
        caps.add("multi-gpu")
    return caps


def parse_node_profile(user_message: str):
    """Reconstruct ``(stage_raw, NodeProfile)`` from a select_skills user_message.

    ``NodeProfile.stage`` is canonicalised via ``STAGE_MAP``. Hardware is parsed
    from a ``Compute:`` line if present (ABSENT in spike-018 -> all-None UNKNOWN,
    which is correct). Sections not present in the record stay ``None``.
    """
    if not isinstance(user_message, str):
        user_message = ""
    m = _STAGE_RE.search(user_message)
    stage_raw = m.group(1) if m else "unknown"

    cm = _COMPUTE_RE.search(user_message)
    if cm:
        raw = cm.group(1).strip()
        if raw.endswith("."):
            raw = raw[:-1]
        hw = parse_hardware(raw)
    else:
        hw = dict(_UNKNOWN_HW)

    # Locate each present section header and slice its body up to the next header.
    found = []
    for marker, field in _SECTION_MARKERS:
        mm = re.search(r"(?m)^" + re.escape(marker) + r"[ \t]*$", user_message)
        if mm:
            found.append((mm.start(), mm.end(), field))
    found.sort()
    sections: dict[str, str] = {}
    for i, (_s, e, field) in enumerate(found):
        nxt = found[i + 1][0] if i + 1 < len(found) else len(user_message)
        sections[field] = user_message[e:nxt].strip()

    profile = NodeProfile(
        stage=STAGE_MAP.get(stage_raw, stage_raw),
        task_text=sections.get("task_text", ""),
        parent_code=sections.get("parent_code"),
        parent_error=sections.get("parent_error"),
        parent_analysis=sections.get("parent_analysis"),
        hardware=hw,
        runtime_capabilities=_runtime_caps(hw),
    )
    return stage_raw, profile


def reconstruct_profiles(traj_dir: Path) -> list[dict]:
    """Reconstruct one node record per ``select_skills`` ask in the trajectory's
    prompts.jsonl. Each: ``{stage_raw, stage, profile, recorded_skills}`` where
    ``recorded_skills`` is the legacy selector's ACTUAL recorded pick (for the
    legacy-exposure baseline; empty list = the legacy selector declined)."""
    nodes: list[dict] = []
    for rec in rss._read_jsonl(Path(traj_dir) / "prompts.jsonl"):
        if rec.get("func_spec_name") != "select_skills":
            continue
        stage_raw, profile = parse_node_profile(rec.get("user_message") or "")
        sels = rss._parse_selector_output(rec.get("output"))
        recorded = sorted({s["skill_name"] for s in sels if s.get("skill_name")})
        nodes.append({
            "stage_raw": stage_raw,
            "stage": profile.stage,
            "profile": profile,
            "recorded_skills": recorded,
        })
    return nodes


# --------------------------------------------------------------------------- #
# Deterministic linker replay (hard filter -> budget cap -> closure). NO LLM.
# --------------------------------------------------------------------------- #

def deterministic_link(profile, cap_skills, budget) -> dict:
    """Run the deterministic linker layers only, stubbing the LLM selector to
    admit all survivors up to ``budget.max_units`` (then dependency closure).

    Returns the survivor set, the hard-filter rejections, and the deterministic
    admitted set (first ``max_units`` survivors in registry order + closure). No
    selector runs. NOTE: ``max_units`` caps the number of ROOTS (before closure),
    not the total injected count — a different LLM-chosen root subset could pull
    in more dependencies. So this bounds the ROOT count at max_units; the total
    injected count is roots + their dependency closure, which the live LLM path
    can exceed for the same node. It is NOT a strict upper bound on injected units."""
    units, by_qid = cl._build_registry(cap_skills)
    survivors, rejected = cl._hard_filter(units, by_qid, profile)
    survivor_qids = [r["qid"] for r in survivors]

    # Budget cap is applied BEFORE closure (HLD §10.2-D). With no selector to rank
    # survivors, take them in deterministic registry order.
    roots = survivor_qids[: budget.max_units]
    order, dep_added, dep_dropped = cl._dep_closure(roots, by_qid, rejected)

    # Assertion-style hard-constraint violation check: every ADMITTED unit must
    # still pass its own intrinsic constraints against this profile (~0 by
    # construction — survivors passed the filter, closure only adds non-rejected
    # deps). A non-empty list here would be a filter bug.
    violations = [
        qid for qid in order
        if cl._intrinsic_reason(by_qid[qid], profile) is not None
    ]

    return {
        "loaded_units": len(units),
        "survivors": survivor_qids,
        "survivor_count": len(survivor_qids),
        "rejected": dict(rejected),  # qid -> reason
        "admitted_order": order,     # capability upper-bound rendered set
        "cap_upper_bound": len(order),
        "dep_added": list(dep_added),
        "dep_dropped": list(dep_dropped),
        "violations": violations,
    }


def legacy_exposure(recorded_skills, units_by_skill: dict[str, int]) -> int:
    """Legacy exposure for a node = number of capability units in the packages the
    legacy selector ACTUALLY loaded (a package exposes ALL its units). An
    empty/declined recorded selection exposes 0 (HLD §12-E2 / plan)."""
    return sum(units_by_skill.get(name, 0) for name in recorded_skills)


# --------------------------------------------------------------------------- #
# Optional --with-selector LLM path (OFF by default; sequential foreground).
# --------------------------------------------------------------------------- #

def make_claude_selector(model: str = "", timeout: int = 180):
    """Best-effort ``select_capabilities`` transport over the ``claude`` CLI
    (Claude subscription; MLEVAL_LLM_TRANSPORT=claude style). Returns an
    ``llm_query(system, user, func_spec) -> dict`` for ``capability_linker.link``.

    CAVEAT (documented, why this is OFF by default): this fires ONE synchronous,
    foreground ``claude -p`` call PER NODE — 80 spike-018 nodes = 80 sequential
    subscription calls, minutes-to-tens-of-minutes wall clock, and nondeterministic
    output. It is provided so the tighter (post-selection) capability exposure can
    be measured as the E2 follow-up; the DEFAULT replay makes NO LLM calls."""
    import os
    import subprocess

    def llm_query(system_message, user_message, func_spec):
        schema = getattr(func_spec, "json_schema", None)
        schema_txt = json.dumps(schema) if schema else (
            '{"selections":[{"capability_id":"skill/unit","reason":"...",'
            '"reference_files":[]}],"decline_reason":"..."}'
        )
        prompt = (
            f"{system_message}\n\n{user_message}\n\n"
            "Return ONLY a single JSON object (no prose, no code fence) matching "
            f"this schema:\n{schema_txt}\n"
            'Use "capability_id" values exactly as listed above; return an empty '
            '"selections" list with a "decline_reason" if none apply.'
        )
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        if model:
            cmd[3:3] = ["--model", model]
        try:
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            raise RuntimeError(f"claude transport failed: {e}") from e
        out = (r.stdout or "").strip()
        mo = re.search(r"\{.*\}", out, re.DOTALL)
        if not mo:
            return {"selections": [], "decline_reason": "selector returned no JSON"}
        try:
            return json.loads(mo.group(0))
        except json.JSONDecodeError:
            return {"selections": [], "decline_reason": "selector JSON parse failed"}

    return llm_query


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _stats(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "sum": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "sum": sum(xs),
        "mean": round(statistics.mean(xs), 4),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def replay(root: Path, cap_skills: list[dict], budget, with_selector: bool,
           selector_model: str = "") -> dict:
    """Reconstruct every node under ``root`` and run the deterministic replay
    (plus the optional LLM selector). Returns a machine-readable report dict."""
    units_by_skill = {s["name"]: len(s.get("capabilities") or []) for s in cap_skills}
    total_units = sum(units_by_skill.values())

    llm_query = make_claude_selector(model=selector_model) if with_selector else None

    node_rows: list[dict] = []
    traj_meta: list[dict] = []
    for traj in rss.trajectory_dirs(Path(root)):
        nodes = reconstruct_profiles(traj)
        if not nodes:
            traj_meta.append({"dir": traj.name, "nodes": 0})
            continue
        traj_meta.append({"dir": traj.name, "nodes": len(nodes)})
        for nd in nodes:
            det = deterministic_link(nd["profile"], cap_skills, budget)
            row = {
                "trajectory": traj.name,
                "stage_raw": nd["stage_raw"],
                "stage": nd["stage"],
                "hardware_known": nd["profile"].hardware.get("gpu_count") is not None
                or bool(nd["profile"].hardware.get("gpu_name")),
                "recorded_skills": nd["recorded_skills"],
                "recorded_declined": not nd["recorded_skills"],
                "survivor_count": det["survivor_count"],
                "rejected_reasons": [det["rejected"][q] for q in det["rejected"]],
                "cap_upper_bound": det["cap_upper_bound"],
                "dep_added": len(det["dep_added"]),
                "dep_dropped": len(det["dep_dropped"]),
                "legacy_exposure": legacy_exposure(nd["recorded_skills"], units_by_skill),
                "violations": det["violations"],
            }
            if with_selector:
                res = cl.link(nd["profile"], cap_skills, llm_query, budget)
                row["selector_selected_ids"] = res.telemetry.get("selector_selected_ids", [])
                row["cap_actual_exposure"] = len(res.selected_ids)
                row["injected_chars"] = res.telemetry.get("injected_chars", 0)
                row["decline_reason"] = res.telemetry.get("decline_reason")
                row["selector_error"] = res.telemetry.get("selector_error")
            node_rows.append(row)

    return _summarize(node_rows, traj_meta, cap_skills, units_by_skill, total_units,
                      budget, with_selector, root)


def _summarize(rows, traj_meta, cap_skills, units_by_skill, total_units,
               budget, with_selector, root) -> dict:
    stages = ["generate", "refine", "debug", "explore"]
    per_stage: dict[str, dict] = {}

    def _reason_counter(subset) -> dict:
        c = Counter()
        for r in subset:
            c.update(r["rejected_reasons"])
        return {k: c.get(k, 0) for k in REASON_CODES if c.get(k)}

    for st in stages:
        subset = [r for r in rows if r["stage"] == st]
        if not subset:
            continue
        entry = {
            "node_count": len(subset),
            "candidates_survivors": _stats([r["survivor_count"] for r in subset]),
            "hard_filtered_by_reason": _reason_counter(subset),
            "legacy_exposure_units": _stats([r["legacy_exposure"] for r in subset]),
            "capability_upper_bound_units": _stats([r["cap_upper_bound"] for r in subset]),
            "dep_added_total": sum(r["dep_added"] for r in subset),
            "dep_dropped_total": sum(r["dep_dropped"] for r in subset),
            "recorded_declined": sum(int(r["recorded_declined"]) for r in subset),
        }
        if with_selector:
            entry["capability_actual_exposure_units"] = _stats(
                [r.get("cap_actual_exposure", 0) for r in subset])
        per_stage[st] = entry

    # Any non-canonical stages (should be none) get bucketed for completeness.
    other = [r for r in rows if r["stage"] not in stages]
    total_violations = sum(len(r["violations"]) for r in rows)
    nodes_with_violation = sum(1 for r in rows if r["violations"])

    exposure = {
        "legacy_all_units": _stats([r["legacy_exposure"] for r in rows]),
        "capability_upper_bound": _stats([r["cap_upper_bound"] for r in rows]),
        "caveat": (
            "capability_upper_bound is survivors capped by budget.max_units + "
            "dependency closure with the LLM selector STUBBED to admit-all — an "
            "UPPER BOUND, not the injected set. On nodes where the legacy selector "
            "declined (legacy exposure 0), the upper bound is >0 because the "
            "semantic selector that would also likely decline is not run. The real "
            "relative-reduction number (plan gate >=30%) needs the LLM pass "
            "(--with-selector); it is a documented follow-up, not computed here."
        ),
    }
    if with_selector:
        exposure["capability_actual"] = _stats([r.get("cap_actual_exposure", 0) for r in rows])
        leg_sum = exposure["legacy_all_units"]["sum"]
        act_sum = exposure["capability_actual"]["sum"]
        exposure["actual_reduction_vs_legacy_pct"] = (
            round(100.0 * (leg_sum - act_sum) / leg_sum, 2) if leg_sum else None)

    result = {
        "setup": {
            "harness": "replay_capability_linker",
            "source_dir": str(root),
            "deterministic": True,
            "llm_calls": bool(with_selector),
            "budget": {
                "max_units": budget.max_units,
                "max_reference_files": budget.max_reference_files,
                "max_chars": budget.max_chars,
            },
            "capability_schema_version": _safe_schema_version(),
        },
        "library": {
            "skills": [
                {"name": s["name"], "units": len(s.get("capabilities") or []),
                 "has_references": bool(s.get("references"))}
                for s in cap_skills
            ],
            "total_units": total_units,
            "units_by_skill": units_by_skill,
        },
        "trajectories": traj_meta,
        "total_nodes": len(rows),
        "per_stage": per_stage,
        "other_stage_nodes": len(other),
        "exposure": exposure,
        "hard_constraint_violation": {
            "nodes": len(rows),
            "nodes_with_violation": nodes_with_violation,
            "total_violating_units": total_violations,
            "rate": (nodes_with_violation / len(rows)) if rows else 0.0,
            "note": (
                "Assertion check: every admitted unit re-passes its intrinsic hard "
                "constraints against the node profile. 0 by construction validates "
                "the three-valued filter; any non-zero value is a filter defect."
            ),
        },
        "recorded_legacy_selection": {
            "nonempty_nodes": sum(1 for r in rows if not r["recorded_declined"]),
            "declined_nodes": sum(1 for r in rows if r["recorded_declined"]),
            "picks_by_skill": _picks_by_skill(rows),
        },
    }
    if with_selector:
        result["selector_health"] = _selector_health(rows)
    return result


def _picks_by_skill(rows) -> dict:
    c = Counter()
    for r in rows:
        for name in r["recorded_skills"]:
            c[name] += 1
    return dict(c)


# Sentinel decline_reasons that indicate a MASKED transport/parse failure — the
# subprocess ran (or the wrapper produced these) but the output was unusable, so
# the node looks like an abstention (0 units) without the selector actually
# having chosen to inject nothing. These MUST NOT be counted as genuine declines.
_FAILURE_DECLINES = {"selector returned no JSON", "selector JSON parse failed",
                     "selector_returned_none", "selector_skipped"}


def _selector_health(rows) -> dict:
    """Bucket every --with-selector node into selected / genuine-decline /
    no-candidates / masked-failure, so a silent transport failure (which also
    yields 0 exposure) can't be mistaken for real abstention. The exposure
    reduction is only trustworthy if masked_failure == 0."""
    selected = genuine_decline = no_candidates = transport_error = parse_failure = 0
    for r in rows:
        if r.get("selector_error"):
            transport_error += 1
            continue
        dr = r.get("decline_reason")
        picked = bool(r.get("selector_selected_ids"))
        if picked:
            selected += 1
        elif dr == "no_candidates":
            no_candidates += 1
        elif dr in _FAILURE_DECLINES:
            parse_failure += 1
        else:
            genuine_decline += 1
    masked = transport_error + parse_failure
    return {
        "selected": selected,
        "genuine_decline": genuine_decline,
        "no_candidates": no_candidates,
        "transport_error": transport_error,
        "parse_failure": parse_failure,
        "masked_failure": masked,
        "trustworthy": masked == 0,
        "note": (
            "masked_failure (transport_error + parse_failure) counts nodes whose 0-unit "
            "exposure is a FAILURE, not a real abstention. The exposure-reduction number "
            "is only valid when masked_failure == 0; any non-zero value means some "
            "'reduction' is a silent claude-CLI failure and the run must be redone."
        ),
    }


def _safe_schema_version():
    try:
        return cl._schema_version()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Text report
# --------------------------------------------------------------------------- #

def _fmt(v, spec=".2f"):
    return format(v, spec) if isinstance(v, float) else ("—" if v is None else str(v))


def print_report(report: dict) -> None:
    setup = report["setup"]
    lib = report["library"]
    print(f"\n=== Capability-linker replay ({setup['source_dir']}) ===")
    lib_desc = ", ".join(f"{s['name']}({s['units']}u)" for s in lib["skills"])
    print(f"library: {lib_desc} | total {lib['total_units']} units "
          f"| budget max_units={setup['budget']['max_units']}")
    print(f"deterministic={setup['deterministic']} llm_calls={setup['llm_calls']} "
          f"| total nodes {report['total_nodes']}\n")

    print("| stage | nodes | survivors(mean) | legacy-units(mean) | "
          "cap-UPPER-BOUND(mean) | dep+ | dep- | declined |")
    print("|---|---|---|---|---|---|---|---|")
    for st, e in report["per_stage"].items():
        print(f"| {st} | {e['node_count']} | "
              f"{_fmt(e['candidates_survivors']['mean'])} | "
              f"{_fmt(e['legacy_exposure_units']['mean'])} | "
              f"{_fmt(e['capability_upper_bound_units']['mean'])} | "
              f"{e['dep_added_total']} | {e['dep_dropped_total']} | "
              f"{e['recorded_declined']} |")

    exp = report["exposure"]
    print(f"\nExposure (all nodes): legacy-all-units  sum={exp['legacy_all_units']['sum']} "
          f"mean={_fmt(exp['legacy_all_units']['mean'])} median={exp['legacy_all_units']['median']}")
    print(f"                      cap-upper-bound   sum={exp['capability_upper_bound']['sum']} "
          f"mean={_fmt(exp['capability_upper_bound']['mean'])} "
          f"median={exp['capability_upper_bound']['median']}")
    if "capability_actual" in exp:
        a = exp["capability_actual"]
        print(f"                      cap-ACTUAL(LLM)   sum={a['sum']} "
              f"mean={_fmt(a['mean'])} median={a['median']}")
        red = exp.get("actual_reduction_vs_legacy_pct")
        print(f"  ACTUAL exposure reduction vs legacy: "
              f"{_fmt(red)}%  (gate >=30%)")
    print(f"  UPPER-BOUND caveat: {exp['caveat']}")

    sh = report.get("selector_health")
    if sh:
        verdict = "TRUSTWORTHY" if sh["trustworthy"] else \
            f"UNTRUSTWORTHY ({sh['masked_failure']} masked failures)"
        print(f"\nSelector health [{verdict}]: selected={sh['selected']} "
              f"genuine_decline={sh['genuine_decline']} no_candidates={sh['no_candidates']} "
              f"| transport_error={sh['transport_error']} parse_failure={sh['parse_failure']}")

    hcv = report["hard_constraint_violation"]
    status = "PASS (0 violations)" if hcv["total_violating_units"] == 0 else \
        f"FAIL ({hcv['total_violating_units']} violating units)"
    print(f"\nHard-constraint violation: {status} — "
          f"{hcv['nodes_with_violation']}/{hcv['nodes']} nodes.")

    rls = report["recorded_legacy_selection"]
    print(f"Recorded legacy selection: {rls['nonempty_nodes']} nonempty / "
          f"{rls['declined_nodes']} declined; picks {rls['picks_by_skill']}")

    # Hard-filter reason totals across all stages.
    total_reasons = Counter()
    for e in report["per_stage"].values():
        total_reasons.update(e["hard_filtered_by_reason"])
    if total_reasons:
        print("Hard-filtered by reason (all stages): " +
              ", ".join(f"{k}={v}" for k, v in total_reasons.items()))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", type=Path, help="Trajectory dir or run dir (e.g. pulled-results/spike-018)")
    p.add_argument("--library", choices=["built", "all"], default="built",
                   help="built=two E1 built skills (default); all=+5 corpus distractors")
    p.add_argument("--max-units", type=int, default=LinkBudget().max_units,
                   help=f"budget.max_units cap before closure (default {LinkBudget().max_units})")
    p.add_argument("--max-refs", type=int, default=LinkBudget().max_reference_files,
                   help="budget.max_reference_files (only affects --with-selector rendering)")
    p.add_argument("--with-selector", action="store_true",
                   help="OPTIONAL: run the REAL claude-CLI selector per node (LLM calls, "
                        "sequential foreground, nondeterministic). OFF by default; the "
                        "deterministic replay makes NO LLM calls.")
    p.add_argument("--selector-model", default="",
                   help="claude -p model for --with-selector (default: CLI default)")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = p.parse_args(argv)

    if not args.dir.exists():
        print(f"ERROR: {args.dir} does not exist", file=sys.stderr)
        return 1

    cap_skills = build_library(args.library)
    budget = LinkBudget(max_units=args.max_units, max_reference_files=args.max_refs)
    report = replay(args.dir, cap_skills, budget, args.with_selector, args.selector_model)

    if report["total_nodes"] == 0:
        print("No select_skills records found under the given dir (a without-skill "
              "cell, or no prompts.jsonl). 0 usable nodes.", file=sys.stderr)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        return 1

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
