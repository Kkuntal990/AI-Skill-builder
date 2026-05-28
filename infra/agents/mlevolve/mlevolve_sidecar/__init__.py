"""MLEvolve sidecar — monkey-patches applied at import time.

Import this package BEFORE MLEvolve so the patches take effect before any
LLM call or RNG-using code runs. Order matters:

    1. seed                — pins random/numpy/torch from $SEED
    2. openai_apikey_env   — backfill api_key from $OPENAI_API_KEY
    3. prompt_logger       — captures (system, user, output, tokens) per
                              LLM call into $MLEVAL_PROMPTS_LOG
    4. prompt_overlay      — replaces hardcoded persona / impl_guideline /
                              reviewer "submission.csv" fact from the
                              per-task YAML at $MLEVOLVE_PROMPT_OVERLAY

Each submodule applies its patch on import; the order matters because
prompt_logger wraps the LLM call site whose import would otherwise have
already cached the unpatched reference. prompt_overlay must run before
``agents.*`` modules import their dependencies (run_mlevolve.py already
imports this package first).
"""

from . import seed                # noqa: F401
from . import openai_apikey_env   # noqa: F401
from . import prompt_logger       # noqa: F401
from . import prompt_overlay      # noqa: F401
