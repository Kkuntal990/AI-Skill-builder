"""Per-task prompt overlay — three monkey-patches on MLEvolve internals.

Why a per-task overlay (vs forking upstream further):
  MLEvolve hardcodes prompts that assume Kaggle-shaped tasks: a "Kaggle
  Grandmaster" persona, "split into train/val sets" instructions, and a
  code-reviewer environment fact that says ``submission.csv`` is the
  output channel. Our PEFT and script-optimization tasks have different
  contracts (HF pre-split data, stdout metric, no submission file).

  Forking those prompts means re-merging on every upstream bump and
  hardcoding per-task. An overlay (one YAML per task, monkey-patched at
  sidecar import time) keeps upstream pristine, makes new task contracts
  declarative, and falls through to upstream behavior when no overlay is
  set.

What this patches (MVP — 3 surfaces):
  1. ``agents.planner.base_planner.build_chat_prompt_for_model`` — the
     final prompt assembler called by 7 agent files. Intercept the
     ``introduction`` arg and replace with overlay persona if set.
  2. ``agents.prompts.impl_guideline.get_impl_guideline_from_agent`` —
     returns the "Implementation guideline" dict spliced into every
     codegen prompt. Replace its content if overlay instructions are set.
  3. ``agents.prompts.validation_template_prompts.get_code_review_guidelines``
     — the reviewer's "Environment Facts" list, where the
     ``submission.csv`` line lives. Splice in overlay output_location.

Dual-bind invariant:
  Each upstream module has TWO bindings to the patched callable: one in
  the defining submodule and one re-exported from the package
  ``__init__.py``. Agents import via the re-export
  (``from agents.planner import build_chat_prompt_for_model``), so
  patching only the defining submodule leaves the re-export pointing at
  the original. Patches in this file rebind BOTH locations and the
  build-time smoke (``_smoke_imports.py``) asserts the identity holds.

Order of operations:
  ``mlevolve_sidecar/__init__.py`` imports this module. We eagerly import
  ``agents.planner.base_planner`` etc., apply the patches, AND ensure the
  package-level names point at the patches. Later, when MLEvolve's
  ``run.py`` triggers ``from agents.planner import ...`` in each agent
  file, the bound name resolves to our wrapper.

Out of scope for MVP (defer until a task needs them):
  - ``omit_fragments`` for skipping hardcoded blocks
  - ``stepwise_mode: false`` toggle (our config already disables stepwise)
  - ``allowed_packages_extra``
  - per-step persona overrides in ``agents/coder/stepwise_coder.py``

Known residual ignored for MVP:
  - ``agents/result_parse_agent.py:153`` still has a hardcoded "Kaggle
    grandmaster" persona for the result parser. It pollutes prompts.jsonl
    but does not affect generated code.
"""
from __future__ import annotations

import logging

from .overlay_schema import Overlay, load_overlay
from . import skill_retriever

logger = logging.getLogger(__name__)

# Module-level overlay; loaded once at import time. Tests/smoke may call
# ``reload()`` to swap it without restarting the process.
_OVERLAY: Overlay = load_overlay()


def current_overlay() -> Overlay:
    """Return the active overlay (post-import; may be empty)."""
    return _OVERLAY


def reload(path: str | None = None) -> Overlay:
    """Re-read the overlay (from ``path`` or ``$MLEVOLVE_PROMPT_OVERLAY``).

    Useful for the build-time smoke test (``_smoke_imports.py``) which
    needs to load an example overlay AFTER applying patches.
    """
    global _OVERLAY
    _OVERLAY = load_overlay(path)
    return _OVERLAY


# ---------------------------------------------------------------------------
# Patch 1: persona — intercept build_chat_prompt_for_model
# ---------------------------------------------------------------------------
import agents.planner.base_planner as _bp
import agents.planner as _planner_pkg

_orig_build_chat_prompt = _bp.build_chat_prompt_for_model


def _patched_build_chat_prompt(model_name, introduction, user_prompt, assistant_prefix):
    if _OVERLAY.persona_identity is not None:
        # Full replace — the overlay's identity string becomes the entire
        # system message (intro + framing). Callers downstream don't care
        # about the original "Now let's begin..." phrasing; the user_prompt
        # already contains the task description.
        introduction = _OVERLAY.persona_identity

    # Skill catalog + retrieval (Path A, docs/eval/skill-retrieval-design.md).
    # The skill index is built once at import time from MLEVAL_SKILL_PATH.
    # When present we (1) append the L1 catalog to the introduction so the
    # model always knows what skills are available, and (2) retrieve top-k
    # chunks against the user_prompt for draft/improve/debug stages, with
    # threshold gating so idle turns stay clean.
    idx = skill_retriever.current_index()
    if idx is not None:
        introduction = introduction.rstrip() + "\n\n" + idx.catalog_text()
        stage = skill_retriever.detect_stage(user_prompt)
        if stage in skill_retriever.INJECTION_STAGES:
            chunks = idx.search(user_prompt)
            if chunks:
                block = skill_retriever.render_chunks(chunks)
                user_prompt = skill_retriever.inject_into_user_prompt(user_prompt, block)

    return _orig_build_chat_prompt(model_name, introduction, user_prompt, assistant_prefix)


# Critical dual-bind: rebind in the defining submodule AND the package
# re-export site. Without the second line, agents that did
# ``from agents.planner import build_chat_prompt_for_model`` still see
# the unpatched original.
_bp.build_chat_prompt_for_model = _patched_build_chat_prompt
_planner_pkg.build_chat_prompt_for_model = _patched_build_chat_prompt


# ---------------------------------------------------------------------------
# Patch 2: implementation guideline
# ---------------------------------------------------------------------------
import agents.prompts.impl_guideline as _ig
import agents.prompts as _prompts_pkg

_orig_get_impl_guideline_from_agent = _ig.get_impl_guideline_from_agent


def _patched_get_impl_guideline_from_agent(agent):
    """Replace the Implementation guideline block when overlay is set.

    Must preserve dict shape ``{"Implementation guideline": [list of str]}``
    because callers do ``prompt["Instructions"] |= result`` (dict-merge);
    a missing key would silently drop the section instead of overriding it.
    """
    if _OVERLAY.what_to_produce is None and _OVERLAY.self_check is None:
        return _orig_get_impl_guideline_from_agent(agent)

    lines: list[str] = []
    if _OVERLAY.what_to_produce is not None:
        lines.append("🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):")
        lines.append("")
        for item in _OVERLAY.what_to_produce:
            lines.append(f"• {item}")
        lines.append("")
    if _OVERLAY.self_check is not None:
        lines.append("⚠️  **Self-Check Before Finalizing**:")
        for item in _OVERLAY.self_check:
            lines.append(f"□ {item}")
    return {"Implementation guideline": lines}


_ig.get_impl_guideline_from_agent = _patched_get_impl_guideline_from_agent
_prompts_pkg.get_impl_guideline_from_agent = _patched_get_impl_guideline_from_agent


# ---------------------------------------------------------------------------
# Patch 3: code review guidelines — splice output_location into env facts
# ---------------------------------------------------------------------------
import agents.prompts.validation_template_prompts as _vtp

_orig_get_code_review_guidelines = _vtp.get_code_review_guidelines


def _patched_get_code_review_guidelines():
    """Splice overlay.output_location over the hardcoded submission.csv line.

    The reviewer's environment-facts list claims "submission.csv is TRUTH
    - Do NOT Flag", which overrides our task description's "no submission"
    even when ``no_submission_mode=True``. We rewrite the offending line.
    Single call site in the same module (``get_code_review_prompt``) so
    dual-bind isn't needed.
    """
    guidelines = list(_orig_get_code_review_guidelines())
    if _OVERLAY.output_location is None:
        return guidelines
    needle_prefix = "  • **Submission File Location**"
    replacement = f"  • **Output Location**: {_OVERLAY.output_location}"
    return [replacement if line.startswith(needle_prefix) else line for line in guidelines]


_vtp.get_code_review_guidelines = _patched_get_code_review_guidelines


# ---------------------------------------------------------------------------
# Final log
# ---------------------------------------------------------------------------
if _OVERLAY.is_empty:
    logger.info("[overlay] no overlay active (MLEVOLVE_PROMPT_OVERLAY unset or empty) — upstream prompts in effect")
else:
    logger.info(
        "[overlay] active patches: persona=%s impl_guideline=%s review_facts=%s",
        _OVERLAY.persona_identity is not None,
        (_OVERLAY.what_to_produce is not None) or (_OVERLAY.self_check is not None),
        _OVERLAY.output_location is not None,
    )
