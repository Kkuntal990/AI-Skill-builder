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


def _capture_prompt(args: tuple, kwargs: dict):
    """Return whatever payload the provider actually saw.

    MLEvolve has two call styles:
      - ``query(system_message=..., user_message=..., func_spec=..., ...)``
        — used by code review, result parse, leakage check. kwargs-only.
      - ``generate(prompt, cfg, temperature=..., ...)`` — used by stepwise
        StepAgent + MetaAgent, diff improve, planner (the codegen hot path).
        ``prompt`` is positional or kwarg, contains the full LLM input.

    spike-011 root cause: the original wrapper only captured kwargs
    ``system_message`` / ``user_message``, so ALL ``generate()`` calls
    logged ``None`` for both fields → impossible to verify which patches
    were actually firing. We now capture all three independently.
    """
    system_message = kwargs.get("system_message")
    user_message = kwargs.get("user_message")
    # Prompt: kwarg first, then positional[0] as fallback (generate() uses
    # both styles depending on call site).
    prompt = kwargs.get("prompt")
    if prompt is None and args:
        prompt = args[0]
    return system_message, user_message, prompt


def _wrap(provider_name: str, original):
    def _logged(*args, **kwargs):
        system_message, user_message, prompt = _capture_prompt(args, kwargs)
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
                "prompt": _serialize_output(prompt) if prompt is not None else None,
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
            "prompt": _serialize_output(prompt) if prompt is not None else None,
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
