"""MLEvolve sidecar — monkey-patches applied at import time.

Import this package BEFORE MLEvolve so the patches take effect before any
LLM call or RNG-using code runs. Order matters:

    1. seed                — pins random/numpy/torch from $SEED
    2. openai_apikey_env   — backfill api_key from $OPENAI_API_KEY
    3. prompt_logger       — captures (system, user, output, tokens) per
                              LLM call into $MLEVAL_PROMPTS_LOG
    4. skill_retriever     — builds the BM25 skill index from
                              $MLEVAL_SKILL_PATH at import time (no-op
                              when unset). Consumed by prompt_overlay.
    5. env_overlay         — replaces MLEvolve's hardcoded
                              get_prompt_environment with the actual
                              pinned pip-freeze of relevant libs. Reads
                              $MLEVAL_INSTALLED_PACKAGES_PATH (default
                              /opt/agent/installed_packages.txt) which
                              the entrypoint writes before agent runs.
    6. prompt_overlay      — replaces hardcoded persona / impl_guideline /
                              reviewer "submission.csv" fact from the
                              per-task YAML at $MLEVOLVE_PROMPT_OVERLAY,
                              and (when skill_retriever has an index)
                              appends the L1 catalog + injects retrieved
                              chunks per turn.

Each submodule applies its patch on import; the order matters because
prompt_logger wraps the LLM call site whose import would otherwise have
already cached the unpatched reference. prompt_overlay must run before
``agents.*`` modules import their dependencies (run_mlevolve.py already
imports this package first).
"""

from . import seed                # noqa: F401
from . import openai_apikey_env   # noqa: F401
from . import prompt_logger       # noqa: F401
from . import skill_retriever     # noqa: F401
from . import env_overlay         # noqa: F401
from . import prompt_overlay      # noqa: F401
