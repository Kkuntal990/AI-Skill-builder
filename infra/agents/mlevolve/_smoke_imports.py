"""Build-time smoke: prove that run_mlevolve.py's import order works.

Mirrors the real entrypoint's invocation environment:
    python /workspace/_smoke_imports.py

NOT `python -c` inside /workspace/mlevolve — that's a false negative
because cwd-based imports work there but not under the real script.

This file is COPYed into the image and invoked as the final Dockerfile
step. If any import fails (missing package, wrong sys.path order, etc.),
the build fails — far cheaper than a cluster image pull + run + crash.

We do the same path manipulation as run_mlevolve.py before any imports,
then import every module the real run hits during startup:
  - mlevolve_sidecar (which transitively imports llm.openai)
  - MLEvolve's engine + config
  - our universal mleval analyzer
"""
import os
import sys

# Match run_mlevolve.py path setup exactly.
MLEVOLVE_ROOT = "/workspace/mlevolve"
sys.path.insert(0, MLEVOLVE_ROOT)
sys.path.insert(0, "/workspace")
os.chdir(MLEVOLVE_ROOT)

import mlevolve_sidecar  # noqa: F401,E402 — sidecar must import cleanly

# MLEvolve's startup chain
from engine.executor import Interpreter  # noqa: F401,E402
from engine.search_node import Journal  # noqa: F401,E402
from engine.agent_search import AgentSearch  # noqa: F401,E402
from config import load_cfg  # noqa: F401,E402

# Our universal analyzer chain
from mleval.analyzer import adapter_mlevolve  # noqa: F401,E402
from mleval.analyzer import stage_classifier  # noqa: F401,E402

# -----------------------------------------------------------------------------
# Prompt-overlay regression guards. The overlay's monkey-patches depend on
# specific upstream symbols + a dual-bind invariant. Catching breakage here
# avoids a cluster image pull + 5-min trajectory + crash.
# -----------------------------------------------------------------------------
import agents.planner  # noqa: E402
import agents.planner.base_planner  # noqa: E402
import agents.prompts  # noqa: E402
import agents.prompts.impl_guideline  # noqa: E402
import agents.prompts.validation_template_prompts  # noqa: E402
from mlevolve_sidecar import prompt_overlay  # noqa: E402

# 1. Patch targets exist (catches upstream refactor)
assert callable(agents.planner.build_chat_prompt_for_model), \
    "build_chat_prompt_for_model missing — upstream refactored?"
assert callable(agents.planner.base_planner.build_chat_prompt_for_model), \
    "build_chat_prompt_for_model missing in base_planner"
assert callable(agents.prompts.get_impl_guideline_from_agent), \
    "get_impl_guideline_from_agent missing in agents.prompts"
assert callable(agents.prompts.impl_guideline.get_impl_guideline_from_agent), \
    "get_impl_guideline_from_agent missing in impl_guideline"
assert callable(agents.prompts.validation_template_prompts.get_code_review_guidelines), \
    "get_code_review_guidelines missing in validation_template_prompts"

# 2. Dual-bind invariant: re-exports point at OUR patched fn, not upstream's.
# This is the single most valuable line — catches the most common monkey-patch
# foot-gun (rebinding the submodule but not the package re-export).
assert agents.planner.build_chat_prompt_for_model is prompt_overlay._patched_build_chat_prompt, \
    "DUAL-BIND broken: agents.planner.build_chat_prompt_for_model is not the patched wrapper"
assert agents.planner.base_planner.build_chat_prompt_for_model is prompt_overlay._patched_build_chat_prompt, \
    "DUAL-BIND broken: agents.planner.base_planner.build_chat_prompt_for_model is not the patched wrapper"
assert agents.prompts.get_impl_guideline_from_agent is prompt_overlay._patched_get_impl_guideline_from_agent, \
    "DUAL-BIND broken: agents.prompts.get_impl_guideline_from_agent is not the patched wrapper"
assert agents.prompts.impl_guideline.get_impl_guideline_from_agent is prompt_overlay._patched_get_impl_guideline_from_agent, \
    "DUAL-BIND broken: agents.prompts.impl_guideline.get_impl_guideline_from_agent is not the patched wrapper"
assert agents.prompts.validation_template_prompts.get_code_review_guidelines is prompt_overlay._patched_get_code_review_guidelines, \
    "Single-bind broken: get_code_review_guidelines is not the patched wrapper"

# 3. Round-trip — apply the example overlay and verify it actually changes output.
prompt_overlay.reload("/workspace/mlevolve_sidecar/overlays/peft_rouge.yaml")
ov = prompt_overlay.current_overlay()
assert not ov.is_empty, "peft_rouge.yaml loaded but produced empty overlay"
sample = agents.planner.build_chat_prompt_for_model(
    "gpt-4-test", "🏆 You are a Kaggle Grandmaster - top-tier ML expert", "user msg", "assistant"
)
assert isinstance(sample, dict) and "Kaggle" not in sample["system"], \
    f"Overlay persona did not replace upstream intro: {sample!r}"
# Reset overlay so module import for production doesn't carry the test config.
prompt_overlay.reload(None)
assert prompt_overlay.current_overlay().is_empty, "Reset to upstream-default failed"

print("OK: run_mlevolve.py + MLEvolve + mleval analyzer + prompt_overlay all import cleanly")
