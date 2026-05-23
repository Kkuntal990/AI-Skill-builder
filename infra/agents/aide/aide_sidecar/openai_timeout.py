"""Inject a finite HTTP timeout into the openai client.

AIDE constructs ``openai.OpenAI(...)`` without specifying a timeout. In
practice we observed LLM calls hanging indefinitely on idle TCP connections
(pilot mvp-001: ~24 min stuck on an ESTABLISHED OpenRouter connection with
zero rx/tx, ultimately killed by SIGTERM not by client timeout).

This module patches ``openai.OpenAI.__init__`` and ``openai.AsyncOpenAI.__init__``
to inject ``timeout=$MLEVAL_LLM_TIMEOUT_SEC`` (default 120s, 10s connect) when
the caller didn't pass one. Per-request ``timeout=`` overrides still work.
"""

from __future__ import annotations

import os

import httpx
import openai

_DEFAULT_TIMEOUT_SEC = float(os.environ.get("MLEVAL_LLM_TIMEOUT_SEC", "120"))
_DEFAULT_CONNECT_SEC = float(os.environ.get("MLEVAL_LLM_CONNECT_TIMEOUT_SEC", "10"))

_TIMEOUT = httpx.Timeout(timeout=_DEFAULT_TIMEOUT_SEC, connect=_DEFAULT_CONNECT_SEC)


def _patch_client(cls) -> None:
    orig_init = cls.__init__

    def patched_init(self, *args, timeout=None, **kwargs):
        if timeout is None:
            timeout = _TIMEOUT
        return orig_init(self, *args, timeout=timeout, **kwargs)

    patched_init.__wrapped__ = orig_init  # type: ignore[attr-defined]
    cls.__init__ = patched_init


_patch_client(openai.OpenAI)
_patch_client(openai.AsyncOpenAI)
