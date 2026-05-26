"""Capture every aide.backend.* LLM call to prompts.jsonl.

Why patch provider_to_query_func entries instead of `aide.backend.query`:
`aide.agent` does `from .backend import query` at module import time, which
captures the ORIGINAL `query` reference. By the time our sidecar runs, that
binding is already frozen. But the original `query` looks up
`provider_to_query_func[provider]` at call time, so patching the dict
entries reaches every code path (agent, journal2report, future callers).

This file also disables AIDE's hardcoded `order=["Fireworks"]` in the
openrouter backend — it broke when serving DeepSeek (not on Fireworks).
Preferred path is to set OPENAI_BASE_URL=https://openrouter.ai/api/v1 in
the env so the openai backend is used; this still supports the openrouter
backend as a fallback.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import aide.backend as _backend

_LOG_PATH = Path(os.environ.get("MLEVAL_PROMPTS_LOG", "./prompts.jsonl"))
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _is_null_response(output) -> bool:
    """Detect the OpenRouter/DeepSeek 'empty-completion' pattern.

    Observed in mvp-006: a draft retry returned `output='null', out_tok=1,
    req_time=0.87s` — far too fast to be a real generation, and AIDE's
    extract_code(None) blew up downstream. Retrying once recovers in most
    cases; the upstream cause is OpenRouter pre-filter latency, not the model.
    """
    if output is None:
        return True
    if isinstance(output, str) and output.strip().lower() in ("", "null"):
        return True
    return False


# For deepseek/* on OpenRouter, no single provider serves both AIDE
# call types (probed 2026-05-25):
#
#   call type        working providers
#   ---------------  ---------------------------------------------
#   tool_choice      DeepInfra (ONLY)
#   free-form draft  Novita, AtlasCloud, GMICloud, Venice
#
# DeepInfra content-filters the AIDE Kaggle-grandmaster system prompt
# (~7 KB) and returns a 1-token `null` in ~1.5s — pre-filter rejection,
# not the model. The four free-form providers don't filter but 404 on
# tool_choice with "No endpoints found that support the provided
# tool_choice value". Alibaba/Parasail/AkashML/SiliconFlow/Morph/
# StreamLake/Venice were each ruled out for one reason or the other.
#
# So we split routing per call type below. Non-deepseek models stay on
# OpenRouter's default routing.
_DEEPSEEK_TOOL_CHOICE_PROVIDERS = ["DeepInfra"]
_DEEPSEEK_FREEFORM_PROVIDERS = ["Novita", "AtlasCloud", "GMICloud", "Venice"]


def _apply_provider_routing(model_kwargs: dict, has_func_spec: bool) -> None:
    """Inject deepseek/* provider routing based on call type."""
    model = model_kwargs.get("model") or ""
    if not model.startswith("deepseek/"):
        return
    eb = model_kwargs.setdefault("extra_body", {})
    if not isinstance(eb, dict):
        return  # caller passed something exotic; don't overwrite
    provider = eb.setdefault("provider", {})
    if "only" in provider:
        return  # caller already pinned; respect it
    if has_func_spec:
        provider["only"] = list(_DEEPSEEK_TOOL_CHOICE_PROVIDERS)
    else:
        provider["only"] = list(_DEEPSEEK_FREEFORM_PROVIDERS)


def _make_logged(provider_name: str, original):
    """Return a wrapper around a provider-specific query func that logs each
    call, applies provider routing, and retries once on a null response.
    """

    def _logged(system_message=None, user_message=None, func_spec=None, **model_kwargs):
        _apply_provider_routing(model_kwargs, has_func_spec=func_spec is not None)
        for attempt in (1, 2):
            started = time.time()
            output, req_time, in_tok, out_tok, info = original(
                system_message=system_message,
                user_message=user_message,
                func_spec=func_spec,
                **model_kwargs,
            )
            record = {
                "ts": started,
                "provider": provider_name,
                "model": model_kwargs.get("model"),
                "system_message": system_message,
                "user_message": user_message,
                "output": output if isinstance(output, str) else json.dumps(output, default=str),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "req_time_sec": req_time,
                "func_spec_name": getattr(func_spec, "name", None),
                "attempt": attempt,
            }
            with _LOG_PATH.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            if attempt == 2 or not _is_null_response(output):
                return output, req_time, in_tok, out_tok, info
            # null on attempt 1 — fall through to attempt 2

    return _logged


# Guard against upstream replacing the plain dict with a Mapping/proxy that
# blocks item assignment. Caught early here, before a silent logging gap.
assert isinstance(_backend.provider_to_query_func, dict), (
    "aide.backend.provider_to_query_func is no longer a plain dict — "
    "the sidecar patch needs a new strategy. Saw: "
    f"{type(_backend.provider_to_query_func).__name__}"
)

# Patch every registered provider in-place. New providers added by upstream
# will need a row here. We only intercept dispatch *after* the top-level
# `aide.backend.query` selects a provider — that function reads the dict
# fresh on every call, so we don't need to also patch the dispatcher itself.
for _provider, _orig in list(_backend.provider_to_query_func.items()):
    _backend.provider_to_query_func[_provider] = _make_logged(_provider, _orig)
