"""MLEvolve sidecar — monkey-patches applied at import time.

Import this package BEFORE MLEvolve so the patches take effect before any
LLM call or RNG-using code runs. Order matters:

    1. seed                — pins random/numpy/torch from $SEED
    2. openai_apikey_env   — backfill api_key from $OPENAI_API_KEY
    3. prompt_logger       — captures (system, user, prompt, output, tokens)
                              per LLM call into $MLEVAL_PROMPTS_LOG
                              (NB: captures both ``query`` kwargs AND
                              ``generate`` positional prompt — see prompt_logger.py)
    4. skill_retriever     — loads skills from $MLEVAL_SKILL_PATHS (or
                              $MLEVAL_SKILL_PATH singular for back-compat),
                              patches ``get_prompt_environment`` to splice
                              an "Available Skills" catalog and "Skill
                              Reference" body into ``prompt["Instructions"]``.
                              Reaches stepwise prompts via the dict-copy in
                              ``stepwise_coder``.

Each submodule applies its patch on import. The order ensures
prompt_logger wraps the LLM call site BEFORE MLEvolve's agent modules
cache references to it, and skill_retriever patches
get_prompt_environment BEFORE draft_agent.py imports it at module load.
``run_mlevolve.py`` imports this package first so all patches install
before any agent module loads.

History note (spike-011): we previously also shipped:
  - ``prompt_overlay`` — overrode persona / impl_guideline / code-review
    via per-task YAML. Bypassed by MLEvolve's hardcoded stepwise generation
    path (which doesn't route through ``build_chat_prompt_for_model``).
  - ``env_overlay`` — replaced the upstream 15-package env hint. Decided
    MLE-Bench parity (keep upstream's stock list) is the cleaner baseline.

Both removed in favor of trusting the published MLEvolve config and
limiting our patches to the minimum needed for the A/B treatment.
"""

from . import seed                # noqa: F401
from . import openai_apikey_env   # noqa: F401
from . import prompt_logger       # noqa: F401
from . import token_budget        # noqa: F401  — raises max_tokens default (anti-truncation); AFTER prompt_logger so it wraps outermost
from . import skill_retriever     # noqa: F401
