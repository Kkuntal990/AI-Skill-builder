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

Fallback: library empty -> no catalog, no selector (baseline). Selector raises
-> all skill bodies (SKILL.md only) so the node still gets skill content.
Selector returns [] -> catalog only (the model declined).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys

from . import skill_retriever
from .eval_harness import apply_impl_guideline_harness

logger = logging.getLogger(__name__)

# Sentinels.
_UNSET = object()          # selection not computed yet for this node
_FALLBACK_ALL = object()   # selector errored -> load all skill bodies (no refs)

# agent module fullname -> stage label.
_TARGETS = {
    "agents.draft_agent": "draft",
    "agents.improve_agent": "improve",
    "agents.debug_agent": "debug",
    "agents.evolution_agent": "evolution",
}

_SELECTOR_SPEC = None

# ---------------------------------------------------------------------------
# Selector (Activation + Execution)
# ---------------------------------------------------------------------------

def _get_selector_spec():
    """Build + cache the FunctionSpec. Strict-mode safe (no oneOf)."""
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
            "coder's context. Return an empty list if none apply."
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
                        },
                        "required": ["skill_name", "references"],
                    },
                }
            },
            "required": ["selections"],
        },
    )
    return _SELECTOR_SPEC


def _selector_system(skills) -> str:
    return (
        "You are a skill router for an ML-engineering coding agent. Given the "
        "current coding sub-task, choose which of the available skills (and which "
        "of each skill's reference files) are RELEVANT and should be loaded into "
        "the coder's context. Be selective: pick a skill only if it clearly helps "
        "the current sub-task; pick references conservatively. Return an empty "
        "list if none apply.\n\n"
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
    parts = [f"Stage: {stage or 'unknown'}"]
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


def _run_selector(agent, skills):
    """Return a list of {skill_name, references} dicts, or _FALLBACK_ALL on error."""
    stage = getattr(agent, "_mleval_stage", None)
    parent = getattr(agent, "_mleval_parent", None)
    try:
        import llm
        out = llm.query(
            system_message=_selector_system(skills),
            user_message=_selector_user(agent, stage, parent),
            func_spec=_get_selector_spec(),
            model=agent.acfg.feedback.model,
            temperature=0.0,
            cfg=agent.cfg,
        )
        selections = out.get("selections", []) if isinstance(out, dict) else []
        valid = {s["name"] for s in skills}
        cleaned = []
        for sel in selections:
            if isinstance(sel, dict) and sel.get("skill_name") in valid:
                refs = sel.get("references", [])
                cleaned.append({
                    "skill_name": sel["skill_name"],
                    "references": refs if isinstance(refs, list) else [],
                })
        logger.info(
            "[skill_injector] stage=%s selected=%s",
            stage, [s["skill_name"] for s in cleaned],
        )
        return cleaned
    except Exception as e:  # noqa: BLE001 — never break codegen on selector failure
        logger.warning("[skill_injector] selector failed (%s); fallback=all bodies", e)
        return _FALLBACK_ALL


def _ensure_selection(agent, skills):
    """Run the selector at most once per node; cache the result on the agent."""
    cached = getattr(agent, "_mleval_selection", _UNSET)
    if cached is not _UNSET:
        return cached
    selection = _run_selector(agent, skills)
    try:
        agent._mleval_selection = selection
    except Exception:  # noqa: BLE001 — agent may not accept attrs (defensive)
        pass
    return selection


def _render_selected_bodies(selection, skills) -> list[str]:
    by_name = {s["name"]: s for s in skills}
    blocks: list[str] = []
    if selection is _FALLBACK_ALL:
        return [f"### Skill: {s['name']}\n\n{s['body']}" for s in skills]
    for sel in selection:
        s = by_name.get(sel["skill_name"])
        if s is None:
            continue
        block = f"### Skill: {s['name']}\n\n{s['body']}"
        refs = sel.get("references", [])
        if refs == ["__all__"]:
            chosen = s["reference_files"]
        elif isinstance(refs, list):
            chosen = [r for r in refs if r in s["references"]]
        else:
            chosen = []
        for fn in chosen:
            block += f"\n\n#### references/{fn}\n\n{s['references'][fn]}"
        blocks.append(block)
    return blocks


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
        try:
            return orig_run(agent, *args, **kwargs)
        finally:
            agent._mleval_stage = None
            agent._mleval_parent = None
            agent._mleval_selection = _UNSET

    run._mleval_patched = True
    return run


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
                    bodies = _render_selected_bodies(
                        _ensure_selection(agent, skills), skills
                    )
                    if bodies:
                        gl.append("")
                        gl.append("## Loaded Skill Content")
                        gl.extend(bodies)
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

logger.info("[skill_injector] registered import hook for %d agent modules", len(_TARGETS))
