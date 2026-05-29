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

# -----------------------------------------------------------------------------
# Skill-retriever regression guards. Path A architecture
# (docs/eval/skill-retrieval-design.md): the sidecar builds a BM25 index over
# SKILL.md + references/*.md at import time, and prompt_overlay's wrapper
# injects L1 catalog + retrieved chunks. These checks catch the most likely
# regressions: chunker producing empty output, retriever returning nothing for
# an obviously relevant query, or stage-detection heuristic missing the draft
# marker (would silently skip injection in production).
# -----------------------------------------------------------------------------
from mlevolve_sidecar import skill_retriever  # noqa: E402

# Use the peft-tuning skill bundled in the image as a fixture. If absent we
# skip these assertions (don't fail the build of an image that doesn't ship a
# skill — separate concern).
_SKILL_FIXTURE = "/results/data/peft-tuning"  # PVC path used at runtime
_FALLBACK_FIXTURE = "/workspace/skills/peft-tuning"  # if baked into image
import pathlib as _pl  # noqa: E402

_fixture = next(
    (p for p in (_SKILL_FIXTURE, _FALLBACK_FIXTURE) if _pl.Path(p).is_dir()),
    None,
)
if _fixture is None:
    print("WARN: no peft-tuning skill fixture found at runtime PVC or /workspace/skills — skipping skill_retriever smoke")
else:
    idx = skill_retriever.reload(_fixture)
    assert idx is not None, f"skill_retriever.reload({_fixture}) returned None"
    assert len(idx) >= 3, f"expected ≥3 chunks from peft-tuning, got {len(idx)}"
    assert "peft-tuning" in idx.skill_names, f"skill_name missing: {idx.skill_names}"

    # Catalog rendering: non-empty, contains the skill name
    cat = idx.catalog_text()
    assert "peft-tuning" in cat, "catalog_text() does not mention peft-tuning"
    # Catalog MUST NOT contain backtick fences (spike-006 fix — primed LLM mimicry)
    assert "```" not in cat, "catalog_text() leaked backtick fences — fence-priming will return"
    # Should still include the response-format directive
    assert "Response format" in cat, "catalog_text() missing Response format header"

    # Stage detection: draft marker triggers, generic prompt does not
    assert skill_retriever.detect_stage(
        "Solution sketch guideline\nFine-tune Qwen2.5-3B on SAMSum using LoRA..."
    ) == "draft"
    assert skill_retriever.detect_stage("hello world") is None

    # Retrieval: a clearly-relevant query returns at least one chunk
    chunks = idx.search(
        "fine-tune qwen2.5 with LoRA target_modules for attention layers"
    )
    assert len(chunks) >= 1, "retrieval returned 0 chunks for a clearly-relevant query"
    assert all(c.score > 0 for c in chunks), "retrieved chunks have zero score"

    # Reset for production (the prompt_overlay wrapper consults the current_index)
    skill_retriever.reload(None)
    assert skill_retriever.current_index() is None, "skill_retriever reload(None) did not reset"

# -----------------------------------------------------------------------------
# Env-overlay regression guards (spike-009 → spike-010 fix). Verifies that
# the get_prompt_environment monkey-patch is wired with dual-bind AND reads
# from the env-var-controlled path. Uses a tmpfile fixture so this works at
# build time when /opt/agent/installed_packages.txt doesn't yet exist.
# -----------------------------------------------------------------------------
import os as _os  # noqa: E402
import tempfile as _tempfile  # noqa: E402
from mlevolve_sidecar import env_overlay  # noqa: E402
import agents.prompts.environment as _env_mod  # noqa: E402

# 1. Dual-bind invariant
assert agents.prompts.get_prompt_environment is env_overlay._patched_get_prompt_environment, \
    "DUAL-BIND broken: agents.prompts.get_prompt_environment is not the patched fn"
assert _env_mod.get_prompt_environment is env_overlay._patched_get_prompt_environment, \
    "DUAL-BIND broken: agents.prompts.environment.get_prompt_environment is not the patched fn"

# 2. Round-trip with a tmpfile fixture — verify the file is actually read
with _tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as _tf:
    _tf.write("peft==0.19.1\ntrl==1.5.1\ntransformers==5.9.0\n")
    _fixture_path = _tf.name
_orig_env_path = _os.environ.get(env_overlay.PATH_ENV_VAR)
_os.environ[env_overlay.PATH_ENV_VAR] = _fixture_path
try:
    result = agents.prompts.get_prompt_environment()
    assert isinstance(result, dict) and "Installed Packages" in result, \
        f"get_prompt_environment() shape changed: {result!r}"
    body = result["Installed Packages"]
    assert "peft==0.19.1" in body, f"pinned versions not in body: {body[:300]!r}"
    assert "trl==1.5.1" in body, "trl version missing from body"
    assert "all packages are already installed" not in body.lower(), \
        "upstream lie still present — env_overlay didn't replace"
finally:
    _os.unlink(_fixture_path)
    if _orig_env_path is None:
        _os.environ.pop(env_overlay.PATH_ENV_VAR, None)
    else:
        _os.environ[env_overlay.PATH_ENV_VAR] = _orig_env_path

# 3. Graceful fallback when the file is missing
_os.environ[env_overlay.PATH_ENV_VAR] = "/this/path/does/not/exist"
result_missing = agents.prompts.get_prompt_environment()
assert "missing" in result_missing["Installed Packages"].lower(), \
    "fallback message missing when file absent"
_os.environ.pop(env_overlay.PATH_ENV_VAR, None)

print("OK: run_mlevolve.py + MLEvolve + mleval analyzer + prompt_overlay + skill_retriever + env_overlay all import cleanly")
