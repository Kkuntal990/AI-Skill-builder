"""Pin the metric optimization direction deterministically.

WHY
---
MLEvolve decides whether the eval metric is maximized or minimized with an LLM
call, ``agents.result_parse_agent.determine_metric_direction`` (an unguarded
function-call that fills a ``lower_is_better`` boolean). Observed live
(spike-026): on the IDENTICAL clean gsm8k task at temp 0, the call returned
``lower_is_better=True`` while its own reasoning said "should be maximized" — a
polarity flip on the negated field. Across 4 trajectories on the same task it
was wrong 1/4 times. A flipped direction silently INVERTS the MCGS search (it
then optimizes for WORSE accuracy) and corrupts the whole run.

Every task in our harness has a KNOWN direction (gsm8k exact-match, samsum
ROUGE-L → both maximize). So we pin it instead of letting the LLM guess, the
same principle by which we pin the grader metric and keep description.md clean
(de_kaggle). Set ``MLEVAL_METRIC_MAXIMIZE`` (1/0) in the env; when set we:

  1. replace ``determine_metric_direction`` to set ``agent.metric_maximize``
     from the env deterministically (and skip the LLM call entirely), and
  2. replace ``_validate_metric_direction`` so a per-node parse that
     nondeterministically flips direction does NOT mark the node buggy — it is
     scored with the pinned direction (the per-node LLM opinion is ignored).

When the env var is unset/empty, both wrappers defer to the original MLEvolve
behavior, so vanilla usage is unchanged.

MECHANISM
---------
``engine/agent_search.py`` calls ``result_parse_agent.determine_metric_direction``
and the per-node parser calls ``_validate_metric_direction`` — both as module
globals, resolved at call time. Rebinding them on the ``agents.result_parse_agent``
module object therefore reaches every internal call site. The module is imported
after the sidecar, so we register a ``sys.meta_path`` finder that patches it the
instant it finishes loading (mirrors skill_injector).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sys

logger = logging.getLogger(__name__)

_TARGET = "agents.result_parse_agent"


def _pinned_maximize():
    """True/False if MLEVAL_METRIC_MAXIMIZE is set, else None (not pinned)."""
    v = os.environ.get("MLEVAL_METRIC_MAXIMIZE")
    if v is None or v.strip() == "":
        return None
    return v.strip().lower() not in ("0", "false", "no", "min", "minimize")


def _make_determine(orig):
    if getattr(orig, "_mleval_pinned", False):
        return orig

    def determine_metric_direction(agent):
        m = _pinned_maximize()
        if m is None:
            return orig(agent)
        agent.metric_maximize = m
        agent.metric_maximize_reasoning = (
            "Pinned by harness (MLEVAL_METRIC_MAXIMIZE=%r); LLM "
            "determine_metric_direction skipped to avoid nondeterministic "
            "polarity flips." % os.environ.get("MLEVAL_METRIC_MAXIMIZE")
        )
        logger.info("[metric_direction] PINNED maximize=%s (LLM step skipped)", m)
        return None

    determine_metric_direction._mleval_pinned = True
    return determine_metric_direction


def _make_validate(orig, module):
    if getattr(orig, "_mleval_pinned", False):
        return orig

    def _validate_metric_direction(agent, node, response):
        if _pinned_maximize() is None:
            return orig(agent, node, response)
        # Pin active: trust the global pinned direction. Never mark a node buggy
        # over a per-node LLM direction opinion (which flips the same way).
        try:
            node.metric = module.MetricValue(
                response["metric"], maximize=agent.metric_maximize
            )
            logger.info(
                "[metric_direction] node %s scored with pinned maximize=%s "
                "(per-node direction check bypassed)",
                getattr(node, "id", "?"), agent.metric_maximize,
            )
        except Exception as e:  # noqa: BLE001 — fall back rather than break parsing
            logger.warning("[metric_direction] pinned validate fallback: %s", e)
            return orig(agent, node, response)

    _validate_metric_direction._mleval_pinned = True
    return _validate_metric_direction


def _patch_module(module) -> None:
    if getattr(module, "_mleval_metric_patched", False):
        return
    if _pinned_maximize() is None:
        logger.info("[metric_direction] MLEVAL_METRIC_MAXIMIZE unset; leaving LLM behavior")
        return
    if hasattr(module, "determine_metric_direction"):
        module.determine_metric_direction = _make_determine(module.determine_metric_direction)
    if hasattr(module, "_validate_metric_direction"):
        module._validate_metric_direction = _make_validate(module._validate_metric_direction, module)
    module._mleval_metric_patched = True
    logger.info("[metric_direction] patched %s (pin=%s)", module.__name__, _pinned_maximize())


class _MetricPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        if getattr(loader, "_mleval_metric_wrapped", False):
            return spec
        orig_exec = loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            try:
                _patch_module(module)
            except Exception as e:  # noqa: BLE001
                logger.warning("[metric_direction] post-load patch failed: %s", e)

        loader.exec_module = exec_module
        loader._mleval_metric_wrapped = True
        return spec


def _install() -> None:
    if not any(isinstance(f, _MetricPatchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MetricPatchFinder())
    mod = sys.modules.get(_TARGET)
    if mod is not None:  # already imported (unlikely; sidecar loads first)
        _patch_module(mod)


_install()
