"""Structured per-node skill-selection telemetry → ``$MLEVAL_SELECTION_LOG`` (JSONL).

WHY THIS MODULE EXISTS (separate from prompt_logger.py)
-------------------------------------------------------
``prompts.jsonl`` already captures the raw ``select_skills`` LLM call, but NOT
the *decision the injector made after it* — how the raw output was cleaned,
whether a fallback fired, which caps truncated it, and how many chars of skill
body/reference actually landed in the node's prompt. spike-023 was silent for a
full run because that decision was never persisted: the selector returned ``[]``
six times and the with_skill treatment was near-baseline, invisible until a
post-hoc prompt audit. This module makes the **treatment-empty** condition (and
its opposite, ref-bloat) visible in one ``grep`` over one file.

Two record types, one line each:
  - ``event: "cell_init"``  — one per trajectory at sidecar import. Confirms the
    library size and whether the selector is active. The without_skill cell
    emits ``loaded_skill_count: 0`` so "baseline really loaded zero skills" is a
    positive assertion, not an absence of evidence.
  - ``event: "node_selection"`` — one per codegen node. The selection decision +
    fallback_mode + injected sizes + caps.

Contract: telemetry MUST NOT break codegen. Every write is best-effort; a failed
write is dropped silently (same policy as prompt_logger). The log path is
resolved per write (not cached at import) so tests/replay can retarget it via env.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .version import SIDECAR_VERSION


def _default_path() -> str:
    """Default alongside prompts.jsonl (same trajectory dir on the PVC)."""
    prompts = os.environ.get("MLEVAL_PROMPTS_LOG")
    if prompts:
        return str(Path(prompts).parent / "selection_events.jsonl")
    return "./selection_events.jsonl"


def _log_path() -> Path:
    return Path(os.environ.get("MLEVAL_SELECTION_LOG") or _default_path())


def _ctx() -> dict:
    """Trajectory-identifying context stamped on every record (from the pod env)."""
    return {
        "run_id": os.environ.get("MLEVAL_RUN_ID"),
        "trajectory_id": os.environ.get("MLEVAL_TRAJECTORY_ID"),
        "cell": os.environ.get("CELL"),
        "task": os.environ.get("TASK"),
        "sidecar_version": SIDECAR_VERSION,
    }


def _write(record: dict) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — telemetry must never break codegen
        pass


def log_cell_init(loaded_skill_count: int, library: str, selector_active: bool) -> None:
    """One record per trajectory: does this cell have skills, and is routing on?"""
    _write({
        "ts": time.time(),
        "event": "cell_init",
        **_ctx(),
        "loaded_skill_count": loaded_skill_count,
        "library": library,
        "selector_active": bool(selector_active),
    })


def log_node_selection(**fields) -> None:
    """One record per codegen node: the final injection decision + sizes + caps.

    Callers pass keyword fields (stage, fallback_mode, declined, decline_reason,
    selector_error, selector_model, catalog_skill_count, task_chars_seen,
    selected_skills, selected_references, selection_reasons, injected_body_chars,
    injected_ref_chars, skills_truncated, refs_truncated, cap_max_skills,
    cap_max_refs). Unknown keys are recorded verbatim — this schema is additive.
    """
    rec = {"ts": time.time(), "event": "node_selection", **_ctx()}
    rec.update(fields)
    _write(rec)
