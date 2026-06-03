"""Raise the LLM output-token cap to stop mid-output truncation (spike-012).

ROOT CAUSE: ``llm/openai.py`` defaults ``max_tokens=16384`` for BOTH
``query()`` (func-spec / code-review calls) and ``generate()`` (the
coder / debug calls). deepseek-v4-pro emits large outputs — measured
44K-71K chars in prompts.jsonl, with one code-review call hitting
``out_tokens=16384`` exactly — that get cut off mid-SEARCH/REPLACE-block
or mid-statement. Upstream detects this (``finish_reason == "length"``)
but only logs a warning and returns the partial text; ``generate()``
doesn't check at all. The truncated fragment is then written/applied,
leaving stray ``=======`` markers / unparseable code → SyntaxError. This
is what corrupted the no-skill control 2-for-2, NOT model weakness.

This sidecar bumps the DEFAULT ``max_tokens`` (only when a caller didn't
pass one) so legitimate large outputs aren't clipped. We patch the
provider-level functions in place — same rationale as ``prompt_logger``:
agent modules do ``from llm import query, generate`` and the dispatcher
re-looks-up ``llm.openai.query`` per call, so patching here takes effect
everywhere. Registered AFTER prompt_logger so this wrapper is outermost
(it injects the kwarg; prompt_logger then logs the call as-sent).

Pairs with ``use_diff_mode: True`` (small diff edits rarely approach the
cap). The robust backstop — hard-fail + retry on ``finish_reason ==
"length"`` — is tracked separately; this is the lighter, higher-leverage
half that directly removes the observed truncation.
"""
from __future__ import annotations

import logging

import llm.openai as _openai_provider

logger = logging.getLogger("MLEvolve")

# 16384 was the upstream default. deepseek-v4-pro outputs were observed up
# to ~17K tokens (~71K chars); 32768 gives comfortable headroom without
# being unbounded (a runaway output should still fail loudly, not balloon).
_MAX_TOKENS = 32768

_orig_query = _openai_provider.query
_orig_generate = _openai_provider.generate


def _query(*args, **kwargs):
    # query(system_message, user_message, func_spec=None, cfg=None, **model_kwargs)
    # max_tokens flows via model_kwargs → filtered.get("max_tokens", 16384).
    if kwargs.get("max_tokens") is None:
        kwargs["max_tokens"] = _MAX_TOKENS
    return _orig_query(*args, **kwargs)


def _generate(*args, **kwargs):
    # generate(prompt, cfg, temperature=None, max_tokens=None, ...)
    # max_tokens is the 4th positional param; only inject if the caller
    # didn't pass it positionally or by keyword.
    if len(args) < 4 and kwargs.get("max_tokens") is None:
        kwargs["max_tokens"] = _MAX_TOKENS
    return _orig_generate(*args, **kwargs)


_query._token_budget_patched = True  # type: ignore[attr-defined]
_generate._token_budget_patched = True  # type: ignore[attr-defined]
_openai_provider.query = _query
_openai_provider.generate = _generate

logger.info(f"[token_budget] default max_tokens raised {16384} → {_MAX_TOKENS}")
