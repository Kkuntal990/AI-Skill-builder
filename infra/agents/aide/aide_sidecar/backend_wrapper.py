"""Capture every aide.backend.query call to prompts.jsonl.

AIDE's `backend/__init__.py::query` discards the 5-tuple returned by the
provider-specific query funcs (in_tokens, out_tokens, req_time, info), so
upstream there is no record of prompts or token usage. This wrapper
re-implements the dispatch but persists each call as one JSON line under
$MLEVAL_PROMPTS_LOG (default `./prompts.jsonl`).

Must be imported BEFORE aide.agent / aide.run — those modules do
`from .backend import query` at import time, binding whatever `query` is
visible on `aide.backend` at that moment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import aide.backend as _backend

_LOG_PATH = Path(os.environ.get("MLEVAL_PROMPTS_LOG", "./prompts.jsonl"))
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _wrapped_query(
    system_message,
    user_message,
    model,
    temperature=None,
    max_tokens=None,
    func_spec=None,
    **model_kwargs,
):
    """Mirror of aide.backend.query that logs the full 5-tuple before discarding."""
    started = time.time()

    # Re-implement aide.backend.query's body so we can grab the tuple.
    # See /opt/aide/aide/backend/__init__.py:34-74 (kept in sync there).
    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    provider = _backend.determine_provider(model)
    query_func = _backend.provider_to_query_func[provider]

    compiled_system = (
        _backend.compile_prompt_to_md(system_message) if system_message else None
    )
    compiled_user = (
        _backend.compile_prompt_to_md(user_message) if user_message else None
    )

    output, req_time, in_tok, out_tok, info = query_func(
        system_message=compiled_system,
        user_message=compiled_user,
        func_spec=func_spec,
        **model_kwargs,
    )

    record = {
        "ts": started,
        "model": model,
        "provider": provider,
        "system_message": compiled_system,
        "user_message": compiled_user,
        "output": output if isinstance(output, str) else json.dumps(output, default=str),
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "req_time_sec": req_time,
        "func_spec_name": getattr(func_spec, "name", None),
    }
    with _LOG_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return output


_backend.query = _wrapped_query
