"""MLEvolve sidecar — monkey-patches applied at import time.

Import this package BEFORE MLEvolve so the patches take effect before any
LLM call or RNG-using code runs. Order matters:

    1. seed                — pins random/numpy/torch from $SEED
    2. openai_apikey_env   — backfill api_key from $OPENAI_API_KEY
    3. prompt_logger       — captures (system, user, prompt, output, tokens)
                              per LLM call into $MLEVAL_PROMPTS_LOG
                              (NB: captures both ``query`` kwargs AND
                              ``generate`` positional prompt — see prompt_logger.py)
    4. skill_retriever     — LOADER. Reads a skill library from
                              $MLEVAL_SKILL_LIBRARY (a dir; scans */SKILL.md,
                              skips _-prefixed) or $MLEVAL_SKILL_PATHS /
                              $MLEVAL_SKILL_PATH (back-compat). Exposes
                              loaded_skills() + catalog_text(). No patching.
    5. eval_harness        — RULES ONLY. Task-agnostic benchmark rules +
                              num_workers fix, appended to impl_guideline via
                              skill_injector's wrapper (see eval_harness.py).
    6. skill_injector      — PATCHER (Anthropic progressive disclosure). A
                              sys.meta_path hook rebinds run +
                              get_impl_guideline_from_agent on the 4 codegen
                              agents (draft/improve/debug/evolution): Tier-0
                              catalog into EVERY node, plus a per-node temp-0
                              model selector that loads only the relevant
                              skill(s)+references. Calls eval_harness for the
                              non-skill harness append. Imports LAST so the
                              library is populated and the hook is registered
                              before MLEvolve loads any agent module.

Each submodule applies its patch on import. The order ensures prompt_logger
wraps the LLM call site BEFORE MLEvolve's agent modules cache references to it,
and skill_injector's import hook is registered BEFORE draft_agent.py et al.
load. ``run_mlevolve.py`` imports this package first so all patches install
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
from . import diff_guard          # noqa: F401  — hardens SEARCH/REPLACE patcher vs ======= corruption (ast-guard + divider normalize)
from . import metric_direction    # noqa: F401  — pins maximize/minimize (MLEvolve's LLM determine_metric_direction flips nondeterministically)
from . import skill_retriever     # noqa: F401  — loads the skill library
from . import eval_harness        # noqa: F401  — benchmark rules (no patch; used by skill_injector)
from . import skill_injector      # noqa: F401  — patches the 4 codegen agents (must be LAST)
