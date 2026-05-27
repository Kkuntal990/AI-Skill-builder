"""Capture every MLEvolve LLM call to prompts.jsonl.

Why patch llm.openai.query / llm.openai.generate (not llm.query):
MLEvolve's agent modules do ``from llm import query, generate`` at import
time, which captures a frozen reference to the dispatcher. The dispatcher
(``llm/__init__.py::query``) calls ``llm.openai.query`` or
``llm.gemini.query`` per call. We patch the provider-level functions
in-place so the dispatcher's lookup picks up the wrapped versions on
every call — same pattern that worked for AIDE's ``provider_to_query_func``
dict.

We only patch the OpenAI provider since our spike routes through it
(OpenRouter is OpenAI-compatible). If we later test Gemini we'd add the
same wrap on ``llm.gemini.query``.

Schema written per line of ``prompts.jsonl``:
    ts, provider, model, system_message, user_message, output,
    in_tokens, out_tokens, req_time_sec, func_spec_name

Compatible with the AIDE adapter's reader so ``adapter_mlevolve`` can
reuse the same parsing helpers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import llm.openai as _openai_provider

_LOG_PATH = Path(os.environ.get("MLEVAL_PROMPTS_LOG", "./prompts.jsonl"))
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _serialize_output(output) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except Exception:  # noqa: BLE001
        return str(output)


def _wrap(provider_name: str, original):
    def _logged(*args, **kwargs):
        # MLEvolve's signatures pass system_message and user_message as
        # kwargs almost everywhere; the dispatcher rebuilds model_kwargs
        # before calling the provider. We capture from kwargs since args
        # are mostly tuples of unrelated config.
        system_message = kwargs.get("system_message")
        user_message = kwargs.get("user_message")
        func_spec = kwargs.get("func_spec")
        model = kwargs.get("model")

        started = time.time()
        try:
            result = original(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - started
            _write_record({
                "ts": started,
                "provider": provider_name,
                "model": model,
                "system_message": system_message,
                "user_message": user_message,
                "output": f"<EXCEPTION: {type(exc).__name__}: {exc}>",
                "in_tokens": None,
                "out_tokens": None,
                "req_time_sec": elapsed,
                "func_spec_name": getattr(func_spec, "name", None),
                "exception": True,
            })
            raise

        elapsed = time.time() - started
        # MLEvolve provider funcs return either:
        #   (output, req_time, in_tokens, out_tokens, info) — query()
        #   (output, req_time, in_tokens, out_tokens, info) — generate()
        # We accept both. If it's just a string (some code paths), wrap.
        if isinstance(result, tuple) and len(result) == 5:
            output, req_time, in_tok, out_tok, info = result
        else:
            output, req_time, in_tok, out_tok, info = result, elapsed, None, None, {}

        _write_record({
            "ts": started,
            "provider": provider_name,
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "output": _serialize_output(output),
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "req_time_sec": req_time,
            "func_spec_name": getattr(func_spec, "name", None),
        })
        return result

    return _logged


def _write_record(record: dict) -> None:
    try:
        with _LOG_PATH.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        # Logging failures must not break the agent loop. Silent drop is
        # acceptable here — the prompts.jsonl is best-effort instrumentation.
        pass


_openai_provider.query = _wrap("openai", _openai_provider.query)
_openai_provider.generate = _wrap("openai", _openai_provider.generate)
