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

Per-node injection CAPS (sidecar >= 1.0.0): the selector's picks CAN be bounded to
``MLEVAL_SKILL_MAX_PER_NODE`` skills and ``MLEVAL_SKILL_MAX_REFS_PER_NODE``
references total, but the **default is UNCAPPED** — the selector's chosen refs are
injected in full, as the pre-1.0.0 sidecar did (preserves mvp-032 comparability and
keeps the treatment organic). A finite cap is an explicit ABLATION lever, read at
call time so it can be set via env without a rebuild. Motivation for having the
lever: SkillsBench (arXiv 2602.12670) finds 2–3 injected skills optimal (+18.6pp)
vs 4+ (+5.9pp), comprehensive bundles net-negative (−2.9pp) — but whether a cap
helps *here* is a hypothesis to A/B (uncapped vs e.g. 3/3), not a baked-in default.

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

Delivery-mode dispatch (capability-linker MVP, HLD §10):
  ``MLEVAL_SKILL_DELIVERY_MODE`` (read at call time, default ``legacy``) selects
  the implementation-guideline path. ``legacy`` is the byte-identical control
  described above. ``capability_task`` / ``capability_node`` route through
  ``capability_linker`` instead: the linker builds a compact "## Linked ML
  Capabilities" brief from the skills that carry a validated ``capabilities.json``
  and the current node's search state. The MLEvolve-specific glue is ONLY the
  NodeProfile adapter (``_build_node_profile``) and the ``llm.query`` closure
  (``_make_llm_query``); the linker core is agent-generic. ``capability_task``
  links once at the first (draft/generate) node and reuses the rendered brief
  verbatim thereafter; ``capability_node`` re-links every node. A capability arm
  NEVER silently emits legacy content — on an empty capability library or any
  error it injects nothing (``legacy_fallback`` stays False). The legacy selector
  functions, ``_wrap_run``, and the ``sys.meta_path`` hook are untouched.
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


_UNCAPPED = float("inf")


def _caps() -> tuple[float, float]:
    """(max skills, max references) injected per node, read at call time.

    DEFAULT IS UNCAPPED — the selector's chosen refs are injected in full, exactly
    as the pre-1.0.0 sidecar did. This preserves comparability with mvp-032 and
    keeps the treatment organic. The 3/3 cap is an explicit ABLATION, not the
    baseline: the mvp-032 replay showed 30/32 selections were DELIBERATE explicit
    ref-lists (only 2 were ``__all__``), so a hard cap trims genuine selection
    rather than catching a dump-everything pathology. Whether capping helps is a
    hypothesis to A/B (uncapped vs 3/3), not a default to bake in.

    Value semantics (per env var): unset / empty / negative → uncapped;
    ``0`` → inject none (catalog-only ablation); positive ``N`` → cap at N.
    """
    def _cap(name: str) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return _UNCAPPED
        try:
            v = int(raw)
        except ValueError:
            return _UNCAPPED
        return v if v >= 0 else _UNCAPPED
    return _cap("MLEVAL_SKILL_MAX_PER_NODE"), _cap("MLEVAL_SKILL_MAX_REFS_PER_NODE")


# Capability delivery modes (capability-linker MVP). Any other value — including
# unset / empty / a typo — falls to the legacy control path (the production
# default). Only these two explicit strings enter the capability linker.
_CAPABILITY_MODES = ("capability_task", "capability_node")
_WARNED_MODES: set = set()  # unrecognized delivery-mode values already warned about


def _capability_mode() -> str:
    """The active skill-delivery mode, read at CALL TIME (never cached at import).

    ``MLEVAL_SKILL_DELIVERY_MODE`` ∈ {legacy, capability_task, capability_node};
    unset / empty → ``legacy`` (the production default and A/B control). An
    unrecognised value is NOT treated as a capability mode (see ``_CAPABILITY_MODES``)
    — it stays on the byte-identical legacy path so a config typo never silently
    ships an untested treatment. Because that means a mistyped 'capability' cell
    would run as legacy, we WARN loudly (once per value) on an unrecognized mode.
    """
    mode = os.environ.get("MLEVAL_SKILL_DELIVERY_MODE", "legacy").strip() or "legacy"
    if mode != "legacy" and mode not in _CAPABILITY_MODES and mode not in _WARNED_MODES:
        _WARNED_MODES.add(mode)
        logger.warning(
            "[skill_injector] unrecognized MLEVAL_SKILL_DELIVERY_MODE=%r → running "
            "LEGACY path (valid: legacy, %s)", mode, ", ".join(_CAPABILITY_MODES))
    return mode


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
            # None when uncapped (inf isn't valid JSON); a number when a cap is set.
            cap_max_skills=(None if max_skills == _UNCAPPED else max_skills),
            cap_max_refs=(None if max_refs == _UNCAPPED else max_refs),
        )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Capability delivery (capability-linker MVP — HLD §9-11). The ONLY MLEvolve-
# specific glue for the linker: NodeProfile construction, the llm.query closure,
# and the impl_guideline landing site. The linker core is agent-generic.
# ---------------------------------------------------------------------------

def _build_node_profile(agent):
    """MLEvolve agent state -> agent-generic ``capability_linker.NodeProfile``.

    This adapter is the whole MLEvolve coupling: it maps the native stage vocab
    (draft/improve/debug/evolution) onto the schema's canonical classes via
    STAGE_MAP, reuses ``_task_for_routing`` for the harness-stripped task text,
    reads the parent SearchNode's code/error/analysis defensively, parses the
    factual hardware line, and derives runtime capabilities CONSERVATIVELY —
    ``python`` always, ``gpu`` / ``multi-gpu`` only when the GPU count is KNOWN
    (>0 / >1). Everything else (network, persistent-service, credentials,
    multi-node) is left UNKNOWN (absent from the set), so the linker's hard filter
    reasons three-valued and never claims a capability the sidecar can't verify.
    """
    from . import capability_linker
    raw_stage = getattr(agent, "_mleval_stage", None)
    stage = capability_linker.STAGE_MAP.get(raw_stage, raw_stage)
    task_text = _task_for_routing(getattr(agent, "task_desc", "") or "")
    parent = getattr(agent, "_mleval_parent", None)
    parent_code = getattr(parent, "code", None) if parent is not None else None
    parent_error = getattr(parent, "term_out", None) if parent is not None else None
    parent_analysis = getattr(parent, "analysis", None) if parent is not None else None
    hardware = capability_linker.parse_hardware(_detect_hardware())
    runtime_caps = {"python"}
    gpu_count = hardware.get("gpu_count")
    if gpu_count and gpu_count > 0:
        runtime_caps.add("gpu")
    if gpu_count and gpu_count > 1:
        runtime_caps.add("multi-gpu")
    return capability_linker.NodeProfile(
        stage=stage,
        task_text=task_text,
        parent_code=parent_code,
        parent_error=parent_error,
        parent_analysis=parent_analysis,
        hardware=hardware,
        runtime_capabilities=runtime_caps,
    )


def _make_llm_query(agent):
    """Closure the linker calls as ``llm_query(system, user, func_spec)``.

    Routes to MLEvolve's stock ``llm.query`` at temperature 0 with the agent's
    feedback model + cfg — the same call the legacy selector makes. A raise here
    (transport error / missing attr) propagates into the linker's retry-once →
    decline ladder, so the arm degrades to "inject nothing", never to legacy."""
    import llm

    def _llm_query(system_message, user_message, func_spec):
        return llm.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            model=agent.acfg.feedback.model,
            temperature=0.0,
            cfg=agent.cfg,
        )

    return _llm_query


def _log_capability_node(agent, link_result, mode, cache_reused, cap_skill_count) -> None:
    """Emit one ``capability_node`` telemetry record (HLD §11). Best-effort.

    Ships the linker's telemetry verbatim plus the delivery mode, the live node
    stage (a cached ``capability_task`` result carries the FIRST node's stage in
    ``telemetry['stage']``, so record the current one separately), and whether the
    brief was reused from cache. ``legacy_fallback`` is forced present and False —
    a capability arm must never report legacy content.
    """
    try:
        telem = dict(getattr(link_result, "telemetry", None) or {})
        telem["delivery_mode"] = mode
        telem["cache_reused"] = bool(cache_reused)
        telem.setdefault("loaded_skill_count", cap_skill_count)
        telem.setdefault("legacy_fallback", False)
        from . import capability_linker
        raw_stage = getattr(agent, "_mleval_stage", None)
        telem["node_stage"] = capability_linker.STAGE_MAP.get(raw_stage, raw_stage)
        selection_logger.log_capability_node(**telem)
    except Exception:  # noqa: BLE001 — telemetry must never break codegen
        pass


def _inject_capability(agent, result, mode) -> None:
    """Capability delivery path (HLD §10). Appends the linker's rendered brief to
    the Implementation guideline. NEVER breaks codegen and NEVER falls back to
    legacy content: an empty capability library (the no-skill baseline for these
    modes) or ANY exception injects nothing.

    ``capability_task`` links once at the first (draft/generate) node and caches
    the LinkResult on the shared search agent (``agent._mleval_cap_brief``,
    NOT reset by ``_wrap_run``), reusing the rendered brief at every later node.
    ``capability_node`` re-links fresh for every node using its current state.
    """
    try:
        from . import capability_linker
        cap_skills = skill_retriever.loaded_capability_skills()
        if not cap_skills:
            # An empty capability library is the no-skill baseline (inject nothing) —
            # but record it explicitly so an empty capability TREATMENT is visible in
            # selection_events.jsonl rather than an absence of evidence (spike-023).
            try:
                selection_logger.log_capability_node(
                    delivery_mode=mode,
                    node_stage=capability_linker.STAGE_MAP.get(
                        getattr(agent, "_mleval_stage", None),
                        getattr(agent, "_mleval_stage", None)),
                    loaded_skill_count=len(skill_retriever.loaded_skills()),
                    loaded_capability_count=0,
                    reason="no_valid_manifests",
                    legacy_fallback=False,
                )
            except Exception:  # noqa: BLE001 — telemetry must never break codegen
                pass
            return
        gl = result.get("Implementation guideline")
        if not isinstance(gl, list):
            return

        if mode == "capability_task":
            cached = getattr(agent, "_mleval_cap_brief", _UNSET)
            if cached is not _UNSET:
                link_result, cache_reused = cached, True
            else:
                link_result = capability_linker.link(
                    _build_node_profile(agent), cap_skills, _make_llm_query(agent)
                )
                cache_reused = False
                try:
                    agent._mleval_cap_brief = link_result
                except Exception:  # noqa: BLE001 — agent may reject attrs (defensive)
                    pass
        else:  # capability_node — recompute fresh every node
            link_result = capability_linker.link(
                _build_node_profile(agent), cap_skills, _make_llm_query(agent)
            )
            cache_reused = False

        if link_result.rendered:
            gl.append("")
            gl.append(link_result.rendered)
        _log_capability_node(agent, link_result, mode, cache_reused, len(cap_skills))
    except Exception as e:  # noqa: BLE001 — never break codegen, never fall back to legacy
        logger.warning("[skill_injector] capability injection failed: %s", e)


def _inject_legacy(agent, result) -> None:
    """LEGACY delivery — Tier-0 catalog (always) + per-node selected bodies/refs.

    Byte-for-byte the pre-capability behavior; this is the A/B control and the
    production default. UNCHANGED — do not alter.
    """
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


def _wrap_impl_guideline(orig_fn):
    """Dispatch the impl_guideline seam on the delivery mode (read at call time).

    ``legacy`` runs the unchanged catalog + per-node body/reference selector;
    ``capability_task`` / ``capability_node`` route through the capability linker.
    Both paths run AFTER the benchmark-harness append (which reaches both A/B
    cells identically). The mode is read per call so it can be flipped via env
    without a rebuild and so legacy stays the byte-identical default.
    """
    if getattr(orig_fn, "_mleval_patched", False):
        return orig_fn

    def wrapper(agent):
        result = orig_fn(agent)
        # Benchmark harness (both cells) — see eval_harness.py, not skill content.
        apply_impl_guideline_harness(result)
        mode = _capability_mode()
        if mode in _CAPABILITY_MODES:
            _inject_capability(agent, result, mode)
        else:
            _inject_legacy(agent, result)
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
