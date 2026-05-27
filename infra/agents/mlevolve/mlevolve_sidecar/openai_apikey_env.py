"""Backfill MLEvolve's OpenAI api_key from $OPENAI_API_KEY when YAML is empty.

MLEvolve's llm/openai.py reads cfg.agent.code.api_key directly and passes
it to ``OpenAI(api_key=...)``. We can't put the key in the YAML at render
time (would leak into config artifacts and pod logs); instead we keep
api_key empty and patch the OpenAI client to fall back to env.

The OpenAI client library already falls back to OPENAI_API_KEY when the
api_key argument is None — but MLEvolve passes the empty string, which
the client treats as "user-supplied empty" and rejects. So we patch the
client constructor to coerce "" → None, which then re-enables the env
fallback.
"""
from __future__ import annotations

import openai

_original_init = openai.OpenAI.__init__


def _patched_init(self, *args, **kwargs):
    if kwargs.get("api_key") == "":
        kwargs["api_key"] = None
    if kwargs.get("base_url") == "":
        kwargs["base_url"] = None
    return _original_init(self, *args, **kwargs)


openai.OpenAI.__init__ = _patched_init

# AsyncOpenAI too, in case MLEvolve ever uses it (currently doesn't but
# stays defensive).
try:
    _orig_async = openai.AsyncOpenAI.__init__

    def _patched_async(self, *args, **kwargs):
        if kwargs.get("api_key") == "":
            kwargs["api_key"] = None
        if kwargs.get("base_url") == "":
            kwargs["base_url"] = None
        return _orig_async(self, *args, **kwargs)

    openai.AsyncOpenAI.__init__ = _patched_async
except AttributeError:
    pass
