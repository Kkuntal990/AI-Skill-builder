"""Skill injection — Anthropic progressive disclosure for MLEvolve codegen.

MLEvolve is a single-shot codegen tree-search agent (no tool-use file reads),
so it cannot autonomously read a skill into its context the way Claude Code
does. We stand in for that Read action with a per-node model selector.

Three tiers (matching Anthropic Agent Skills):
  - Discovery  : the L1 catalog (name + description + reference filenames) is
                 spliced into EVERY codegen node via the universal seam
                 ``get_impl_guideline_from_agent`` — the agent is always aware
                 of the whole library.
  - Activation : a temp-0 model selector (``select_skills`` func-call) runs once
                 per node and picks which skill(s) to fully load.
  - Execution  : the same selector picks which ``references/*.md`` to load, so
                 we never dump every skill's full body into every node.

Per-node injection CAPS (sidecar >= 1.0.0): the selector's picks are bounded to
``MLEVAL_SKILL_MAX_PER_NODE`` skills and ``MLEVAL_SKILL_MAX_REFS_PER_NODE``
references total (default 3/3). SkillsBench (arXiv 2602.12670) finds 2–3 injected
skills optimal (+18.6pp) vs 4+ (+5.9pp) and comprehensive bundles net-negative
(−2.9pp); the ref cap also stops a ``["__all__"]`` pick from splicing ~2,300 lines
of references into one prompt. Caps are read at call time so ablations can retune
them via env without a rebuild (0 = inject none — a catalog-only ablation).

Why a sys.meta_path import hook (not an eager patch):
  ``agents/__init__.py`` is empty and the four codegen-agent modules
  (draft/improve/debug/evolution) do not exist when this sidecar imports. Each
  agent does ``from agents.prompts import (... get_impl_guideline_from_agent)``,
  which COPIES the name into the agent module's namespace — so patching the
  definition or the package re-export does NOT change the agent bindings. We
  register a MetaPathFinder that wraps ``exec_module`` for each agent module and
  rebinds ``module.run`` + ``module.get_impl_guideline_from_agent`` the instant
  it finishes loading. ``run_mlevolve.py`` imports this sidecar before MLEvolve,
  so the finder is in place before any agent module loads.

Fallback ladder (selector error): retry once; then load ALL skill bodies (no
references) if the library is small (<= _FALLBACK_ALL_MAX_SKILLS), else fall back
to catalog-only — dumping every body is only safe for a tiny library. A selector
that succeeds but returns ``[]`` is a genuine decline → catalog only. Every path
records a ``fallback_mode`` to selection_events.jsonl (see selection_logger.py).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sys

from . import selection_logger, skill_retriever
from .eval_harness import _detect_hardware, apply_impl_guideline_harness

logger = logging.getLogger(__name__)

# Sentinels.
_UNSET = object()          # selection not computed yet for this node
_FALLBACK_ALL = object()   # selector errored -> load all skill bodies (no refs)

# Selector error → load all bodies only when the library is at most this many
# skills; beyond it, dumping every body distorts the prompt more than it helps
# (distractor regime), so we degrade to catalog-only instead.
_FALLBACK_ALL_MAX_SKILLS = 5

# agent module fullname -> stage label.
_TARGETS = {
    "agents.draft_agent": "draft",
    "agents.improve_agent": "improve",
    "agents.debug_agent": "debug",
    "agents.evolution_agent": "evolution",
}

_SELECTOR_SPEC = None


def _caps() -> tuple[int, int]:
    """(max skills, max references) injected per node. Read at call time so an
    ablation can retune via env without an image rebuild. Value is literal: 0
    injects none (catalog-only), a large number is effectively unlimited.
    Default 3/3. Invalid/empty → default."""
    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name, "")
        try:
            v = int(raw.strip()) if raw.strip() else default
        except ValueError:
            return default
        return v if v >= 0 else default
    return _int("MLEVAL_SKILL_MAX_PER_NODE", 3), _int("MLEVAL_SKILL_MAX_REFS_PER_NODE", 3)


# ---------------------------------------------------------------------------
# Selector (Activation + Execution)
# ---------------------------------------------------------------------------

def _get_selector_spec():
    """Build + cache the FunctionSpec. Strict-mode safe (no oneOf; every declared
    property is required, per OpenAI strict function-calling)."""
    global _SELECTOR_SPEC
    if _SELECTOR_SPEC is not None:
        return _SELECTOR_SPEC
    try:
        from llm import FunctionSpec
    except ImportError:
        from llm.gemini import FunctionSpec
    _SELECTOR_SPEC = FunctionSpec(
        name="select_skills",
        description=(
            "Choose which skills (and which of each skill's reference files) are "
            "relevant to the CURRENT coding sub-task and should be loaded into the "
            "coder's context. Give a one-line reason per chosen skill. Return an "
            "empty list (with a decline_reason) if none apply."
        ),
        json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "skill_name": {"type": "string"},
                            # filenames to load; [] = none (SKILL.md only),
                            # ["__all__"] = every reference.
                            "references": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            # one line: why this skill helps THIS sub-task.
                            "reason": {"type": "string"},
                        },
                        "required": ["skill_name", "references", "reason"],
                    },
                },
                # why nothing applies (empty string when selections is non-empty).
                "decline_reason": {"type": "string"},
            },
            "required": ["selections", "decline_reason"],
        },
    )
    return _SELECTOR_SPEC


def _selector_system(skills) -> str:
    return (
        "You are a skill router for an ML-engineering coding agent. Given the "
        "current coding sub-task, choose which of the available skills (and which "
        "of each skill's reference files) are RELEVANT and should be loaded into "
        "the coder's context. Be selective: pick a skill only if it clearly helps "
        "the current sub-task; pick references conservatively. For each chosen "
        "skill give a one-line reason. Return an empty list — with a "
        "decline_reason — if none apply.\n\n"
        "Available skills (name, description, reference files):\n"
        + skill_retriever.catalog_text()
        + '\n\nFor each chosen skill, set "references" to a list of reference '
        'filenames to load, or ["__all__"] for all of them, or [] for none '
        "(SKILL.md body only)."
    )


# End-of-rules sentinel in infra/tasks/_harness_rules.md (legacy C1 prepend).
from .eval_harness import HARNESS_RULES_MARKER as _HARNESS_RULES_MARKER
# Generous cap: route on the task-specific lead (model/method/data/eval). Must
# comfortably exceed a task instruction's signal-bearing head — the gsm8k
# instruction places LoRA@~2.9k and the batch/left-padding (vLLM) signal@~4.4k
# chars into the task-only text, so 1500 (the old value) cut all of it.
_SELECTOR_TASK_CHARS = 6000


def _task_for_routing(task: str) -> str:
    """Task text the selector routes on: drop the prepended harness-rules header.

    The harness concatenates the constant _harness_rules.md ahead of every
    task's instruction (docs/eval/task-authoring.md C1). Those ~3k chars contain
    submission/validation boilerplate but nothing about the model, method, or
    data — so left in place they push the real task past the truncation window
    and the selector declines every skill (observed spike-023: selections=[]
    x6, treatment silently emptied). Strip them when the sentinel is present;
    otherwise (pre-C1 tasks with no rules header) use the text unchanged.
    """
    idx = task.find(_HARNESS_RULES_MARKER)
    if idx != -1:
        task = task[idx + len(_HARNESS_RULES_MARKER):].lstrip()
    return task


def _selector_user(agent, stage, parent) -> str:
    # Compute line mirrors the factual hardware note the codegen prompt already
    # carries (eval_harness Compute line) so the router can weigh memory-gated
    # advice (e.g. "quantize only if it doesn't fit") against the real GPU/VRAM
    # instead of defaulting to it. Symmetric env fact, never a method hint.
    parts = [f"Stage: {stage or 'unknown'}", f"Compute: {_detect_hardware()}."]
    task = _task_for_routing(getattr(agent, "task_desc", "") or "")
    parts.append(f"Task:\n{task[:_SELECTOR_TASK_CHARS]}")
    if parent is not None:
        term_out = getattr(parent, "term_out", "") or ""
        analysis = getattr(parent, "analysis", "") or ""
        code = getattr(parent, "code", "") or ""
        if stage == "debug":
            if term_out:
                parts.append(f"Error output (tail):\n{term_out[-1500:]}")
            if analysis:
                parts.append(f"Root-cause analysis:\n{analysis[:500]}")
        else:  # improve / evolution
            if code:
                parts.append(f"Current solution code (head):\n{code[:1500]}")
    return "\n\n".join(parts)


def _query_selector(agent, skills):
    """One selector LLM call → (cleaned_selections, decline_reason). Raises on
    LLM/transport error so the caller's retry+fallback ladder can handle it."""
    import llm
    out = llm.query(
        system_message=_selector_system(skills),
        user_message=_selector_user(
            agent, getattr(agent, "_mleval_stage", None),
            getattr(agent, "_mleval_parent", None),
        ),
        func_spec=_get_selector_spec(),
        model=agent.acfg.feedback.model,
        temperature=0.0,
        cfg=agent.cfg,
    )
    selections = out.get("selections", []) if isinstance(out, dict) else []
    decline_reason = out.get("decline_reason") if isinstance(out, dict) else None
    valid = {s["name"] for s in skills}
    cleaned = []
    for sel in selections:
        if isinstance(sel, dict) and sel.get("skill_name") in valid:
            refs = sel.get("references", [])
            cleaned.append({
                "skill_name": sel["skill_name"],
                "references": refs if isinstance(refs, list) else [],
                "reason": sel.get("reason") or "",
            })
    return cleaned, decline_reason


def _run_selector(agent, skills):
    """Return ``(selection, meta)``.

    ``selection`` is a list of ``{skill_name, references, reason}`` dicts, ``[]``
    (declined / catalog-only), or ``_FALLBACK_ALL`` (small-library error path).
    ``meta`` carries the telemetry the injector logs for this node.
    """
    stage = getattr(agent, "_mleval_stage", None)
    task_text = _task_for_routing(getattr(agent, "task_desc", "") or "")
    meta = {
        "selector_model": getattr(
            getattr(getattr(agent, "acfg", None), "feedback", None), "model", None
        ),
        "catalog_skill_count": len(skills),
        "task_chars_seen": min(len(task_text), _SELECTOR_TASK_CHARS),
        "decline_reason": None,
        "selector_error": None,
        "fallback_mode": "none",
    }
    last_exc = None
    for attempt in (1, 2):  # retry once before falling back
        try:
            cleaned, decline_reason = _query_selector(agent, skills)
            if not cleaned:
                meta["fallback_mode"] = "catalog_only"
                meta["decline_reason"] = decline_reason or ""
            logger.info(
                "[skill_injector] stage=%s selected=%s",
                stage, [s["skill_name"] for s in cleaned],
            )
            return cleaned, meta
        except Exception as e:  # noqa: BLE001 — never break codegen on selector failure
            last_exc = e
            logger.warning("[skill_injector] selector attempt %d failed: %s", attempt, e)
    # Both attempts failed — apply the fallback ladder.
    meta["selector_error"] = f"{type(last_exc).__name__}: {last_exc}"
    if len(skills) <= _FALLBACK_ALL_MAX_SKILLS:
        meta["fallback_mode"] = "fallback_all"
        logger.warning("[skill_injector] selector failed; fallback=all bodies (small library)")
        return _FALLBACK_ALL, meta
    meta["fallback_mode"] = "catalog_only_on_error"
    logger.warning("[skill_injector] selector failed; large library → catalog-only")
    return [], meta


def _ensure_selection(agent, skills):
    """Run the selector at most once per node; cache (selection, meta) on the agent."""
    cached = getattr(agent, "_mleval_selection", _UNSET)
    if cached is not _UNSET:
        return cached, getattr(agent, "_mleval_selection_meta", {})
    selection, meta = _run_selector(agent, skills)
    try:
        agent._mleval_selection = selection
        agent._mleval_selection_meta = meta
    except Exception:  # noqa: BLE001 — agent may not accept attrs (defensive)
        pass
    return selection, meta


def _render_selected_bodies(selection, skills, max_skills, max_refs):
    """Render selected skill bodies + references, enforcing per-node caps.

    Returns ``(blocks, stats)``. ``stats`` records exactly what was injected —
    selected_skills, selected_references, per-skill reasons, body/ref char counts,
    and whether the caps truncated skills or refs — so the telemetry reflects the
    prompt as actually built, not the selector's raw ask. The ``_FALLBACK_ALL``
    path loads every body (the small-library error fallback) with no references
    and is not skill-capped (the whole point is "load everything, it's tiny").
    """
    by_name = {s["name"]: s for s in skills}
    blocks: list[str] = []
    stats = {
        "selected_skills": [],
        "selected_references": {},
        "selection_reasons": {},
        "injected_body_chars": 0,
        "injected_ref_chars": 0,
        "skills_truncated": False,
        "refs_truncated": False,
    }
    if selection is _FALLBACK_ALL:
        for s in skills:
            blocks.append(f"### Skill: {s['name']}\n\n{s['body']}")
            stats["selected_skills"].append(s["name"])
            stats["injected_body_chars"] += len(s["body"])
        return blocks, stats

    sel = list(selection)
    if len(sel) > max_skills:
        sel = sel[:max_skills]
        stats["skills_truncated"] = True

    refs_used = 0
    for item in sel:
        s = by_name.get(item["skill_name"])
        if s is None:
            continue
        block = f"### Skill: {s['name']}\n\n{s['body']}"
        stats["selected_skills"].append(s["name"])
        stats["injected_body_chars"] += len(s["body"])
        if item.get("reason"):
            stats["selection_reasons"][s["name"]] = item["reason"]

        refs = item.get("references", [])
        if refs == ["__all__"]:
            candidate = list(s["reference_files"])
        elif isinstance(refs, list):
            candidate = [r for r in refs if r in s["references"]]
        else:
            candidate = []

        chosen: list[str] = []
        for fn in candidate:
            if refs_used >= max_refs:
                stats["refs_truncated"] = True
                break
            chosen.append(fn)
            refs_used += 1
        if chosen:
            stats["selected_references"][s["name"]] = chosen
        for fn in chosen:
            block += f"\n\n#### references/{fn}\n\n{s['references'][fn]}"
            stats["injected_ref_chars"] += len(s["references"][fn])
        blocks.append(block)
    return blocks, stats


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

def _wrap_run(stage, orig_run):
    """Stash stage + parent on the agent and reset the per-node selection cache."""
    if getattr(orig_run, "_mleval_patched", False):
        return orig_run

    def run(agent, *args, **kwargs):
        parent = None
        if stage != "draft":
            parent = kwargs.get("parent_node")
            if parent is None and args:
                parent = args[0]
        agent._mleval_stage = stage
        agent._mleval_parent = parent
        agent._mleval_selection = _UNSET
        agent._mleval_selection_meta = {}
        try:
            return orig_run(agent, *args, **kwargs)
        finally:
            agent._mleval_stage = None
            agent._mleval_parent = None
            agent._mleval_selection = _UNSET
            agent._mleval_selection_meta = {}

    run._mleval_patched = True
    return run


def _log_node_selection(agent, meta, stats, max_skills, max_refs) -> None:
    """Emit one node_selection telemetry record. Best-effort; never raises."""
    try:
        selection_logger.log_node_selection(
            stage=getattr(agent, "_mleval_stage", None),
            fallback_mode=meta.get("fallback_mode"),
            declined=meta.get("fallback_mode") in ("catalog_only", "catalog_only_on_error"),
            decline_reason=meta.get("decline_reason"),
            selector_error=meta.get("selector_error"),
            selector_model=meta.get("selector_model"),
            catalog_skill_count=meta.get("catalog_skill_count"),
            task_chars_seen=meta.get("task_chars_seen"),
            selected_skills=stats.get("selected_skills"),
            selected_references=stats.get("selected_references"),
            selection_reasons=stats.get("selection_reasons"),
            injected_body_chars=stats.get("injected_body_chars"),
            injected_ref_chars=stats.get("injected_ref_chars"),
            skills_truncated=stats.get("skills_truncated"),
            refs_truncated=stats.get("refs_truncated"),
            cap_max_skills=max_skills,
            cap_max_refs=max_refs,
        )
    except Exception:  # noqa: BLE001
        pass


def _wrap_impl_guideline(orig_fn):
    """Append Tier-0 catalog (always) + Tier-1/2 selected bodies to the guideline."""
    if getattr(orig_fn, "_mleval_patched", False):
        return orig_fn

    def wrapper(agent):
        result = orig_fn(agent)
        # Benchmark harness (both cells) — see eval_harness.py, not skill content.
        apply_impl_guideline_harness(result)
        try:
            skills = skill_retriever.loaded_skills()
            if skills:
                gl = result.get("Implementation guideline")
                if isinstance(gl, list):
                    gl.append("")
                    gl.append("## Available Skills (catalog)")
                    gl.append(skill_retriever.catalog_text())
                    selection, meta = _ensure_selection(agent, skills)
                    max_skills, max_refs = _caps()
                    bodies, stats = _render_selected_bodies(
                        selection, skills, max_skills, max_refs
                    )
                    if bodies:
                        gl.append("")
                        gl.append("## Loaded Skill Content")
                        gl.extend(bodies)
                    _log_node_selection(agent, meta, stats, max_skills, max_refs)
        except Exception as e:  # noqa: BLE001 — never break codegen
            logger.warning("[skill_injector] guideline injection failed: %s", e)
        return result

    wrapper._mleval_patched = True
    return wrapper


def _patch_agent_module(module, stage) -> None:
    if getattr(module, "_mleval_skill_patched", False):
        return
    if hasattr(module, "run"):
        module.run = _wrap_run(stage, module.run)
    if hasattr(module, "get_impl_guideline_from_agent"):
        module.get_impl_guideline_from_agent = _wrap_impl_guideline(
            module.get_impl_guideline_from_agent
        )
    module._mleval_skill_patched = True
    logger.info("[skill_injector] patched %s (stage=%s)", module.__name__, stage)


# ---------------------------------------------------------------------------
# Deferred import hook
# ---------------------------------------------------------------------------

class _SkillPatchFinder(importlib.abc.MetaPathFinder):
    """Wrap exec_module for the 4 agent modules so we patch them post-load."""

    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        if getattr(loader, "_mleval_wrapped_exec", False):
            return spec
        orig_exec = loader.exec_module
        stage = _TARGETS[fullname]

        def exec_module(module, _orig=orig_exec, _stage=stage):
            _orig(module)
            try:
                _patch_agent_module(module, _stage)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[skill_injector] post-load patch failed for %s: %s",
                    module.__name__, e,
                )

        loader.exec_module = exec_module
        loader._mleval_wrapped_exec = True
        return spec


def _patch_already_loaded() -> None:
    """Cover the (unlikely) case where an agent module is already imported."""
    for fullname, stage in _TARGETS.items():
        mod = sys.modules.get(fullname)
        if mod is not None:
            _patch_agent_module(mod, stage)
    # Definition + package re-export — belt-and-suspenders for aggregation/fusion
    # (which call the seam but never fire in our config).
    for modname in ("agents.prompts.impl_guideline", "agents.prompts"):
        mod = sys.modules.get(modname)
        fn = getattr(mod, "get_impl_guideline_from_agent", None)
        if fn is not None and not getattr(fn, "_mleval_patched", False):
            mod.get_impl_guideline_from_agent = _wrap_impl_guideline(fn)


# Install at the front so we win the race for the 4 agent modules.
sys.meta_path.insert(0, _SkillPatchFinder())
_patch_already_loaded()

# Cell-init telemetry: record the library size + selector-active state for THIS
# trajectory (both cells). without_skill records loaded_skill_count=0 so
# "baseline loaded zero skills" is a positive assertion in the log.
try:
    _sk = skill_retriever.loaded_skills()
    selection_logger.log_cell_init(
        loaded_skill_count=len(_sk),
        library=os.environ.get("MLEVAL_SKILL_LIBRARY", ""),
        selector_active=bool(_sk),
    )
except Exception:  # noqa: BLE001
    pass

logger.info("[skill_injector] registered import hook for %d agent modules", len(_TARGETS))
