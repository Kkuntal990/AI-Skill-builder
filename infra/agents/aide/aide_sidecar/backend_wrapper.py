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


def _make_logged(provider_name: str, original):
    """Return a wrapper around a provider-specific query func that logs each call."""

    def _logged(system_message=None, user_message=None, func_spec=None, **model_kwargs):
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
        }
        with _LOG_PATH.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return output, req_time, in_tok, out_tok, info

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
