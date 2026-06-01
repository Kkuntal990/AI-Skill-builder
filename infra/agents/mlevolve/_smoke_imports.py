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
# skill_retriever regression guards (Anthropic two-tier injection — post
# spike-011 simplification). Catches: dual-bind drift on
# get_prompt_environment, frontmatter parsing regression, mis-resolved skill
# paths. The fixture path matches the runtime PVC layout.
# -----------------------------------------------------------------------------
import agents.prompts  # noqa: E402
import agents.prompts.environment as _env_mod  # noqa: E402
from mlevolve_sidecar import skill_retriever  # noqa: E402

# 1. Patch target exists (catches upstream refactor)
assert callable(agents.prompts.get_prompt_environment), \
    "get_prompt_environment missing in agents.prompts — upstream refactored?"
assert callable(_env_mod.get_prompt_environment), \
    "get_prompt_environment missing in agents.prompts.environment"

# 2. Dual-bind invariant — both module-level bindings point at OUR patched fn
assert agents.prompts.get_prompt_environment is skill_retriever._patched_get_prompt_environment, \
    "DUAL-BIND broken: agents.prompts.get_prompt_environment is not the patched fn"
assert _env_mod.get_prompt_environment is skill_retriever._patched_get_prompt_environment, \
    "DUAL-BIND broken: agents.prompts.environment.get_prompt_environment is not the patched fn"

# 3. Patched get_prompt_environment returns upstream shape (dict with
#    "Installed Packages" — when no skills loaded, identical to upstream)
_env_result = agents.prompts.get_prompt_environment()
assert isinstance(_env_result, dict) and "Installed Packages" in _env_result, \
    f"get_prompt_environment() shape changed: {_env_result!r}"

# 4. With no MLEVAL_SKILL_PATH(S) set at build time, the patched fn must
#    NOT inject Available Skills / Skill Reference keys (no-op path)
assert "Available Skills" not in _env_result, \
    "skill_retriever leaked Available Skills with no env var set"
assert "Skill Reference" not in _env_result, \
    "skill_retriever leaked Skill Reference with no env var set"

# 5. Fixture round-trip — load the peft-tuning skill from the runtime PVC
#    path or a bundled fallback, then verify catalog + reference render
_SKILL_FIXTURE = "/results/data/peft-tuning"
_FALLBACK_FIXTURE = "/workspace/skills/peft-tuning"
import pathlib as _pl  # noqa: E402
_fixture = next(
    (p for p in (_SKILL_FIXTURE, _FALLBACK_FIXTURE) if _pl.Path(p).is_dir()),
    None,
)
if _fixture is None:
    print(
        "WARN: no peft-tuning skill fixture found at runtime PVC or "
        "/workspace/skills — skipping skill_retriever fixture smoke"
    )
else:
    os.environ["MLEVAL_SKILL_PATHS"] = _fixture
    n = skill_retriever.reload()
    assert n >= 1, f"reload() returned {n} skills, expected >= 1"

    loaded = skill_retriever.loaded_skills()
    assert any(s["name"] == "peft-tuning" for s in loaded), \
        f"peft-tuning skill not in loaded set: {[s['name'] for s in loaded]}"

    # With skill loaded, env dict gains the two new keys
    _env_with_skill = agents.prompts.get_prompt_environment()
    assert "Available Skills" in _env_with_skill, \
        "Available Skills key missing after loading skill"
    assert "Skill Reference" in _env_with_skill, \
        "Skill Reference key missing after loading skill"
    assert "peft-tuning" in _env_with_skill["Available Skills"], \
        "peft-tuning name missing from catalog"
    assert len(_env_with_skill["Skill Reference"]) > 500, \
        "Skill Reference body suspiciously short — references not concatenated?"

    # Reset for production
    os.environ.pop("MLEVAL_SKILL_PATHS", None)
    n = skill_retriever.reload()
    assert n == 0, f"reload() with no env var returned {n} skills, expected 0"

# -----------------------------------------------------------------------------
# prompt_logger regression — capture both query() kwargs and generate()
# positional/kwarg prompt (spike-011 fix). The helper is small enough to
# exercise directly with synthetic args.
# -----------------------------------------------------------------------------
from mlevolve_sidecar import prompt_logger  # noqa: E402

_sm, _um, _p = prompt_logger._capture_prompt(
    args=(), kwargs={"system_message": "sys", "user_message": "usr"}
)
assert _sm == "sys" and _um == "usr" and _p is None, \
    f"query() capture broken: sm={_sm!r} um={_um!r} p={_p!r}"

_sm, _um, _p = prompt_logger._capture_prompt(
    args=({"role": "user", "content": "hi"},), kwargs={}
)
assert _sm is None and _um is None and isinstance(_p, dict), \
    f"generate(positional) capture broken: sm={_sm!r} um={_um!r} p={_p!r}"

_sm, _um, _p = prompt_logger._capture_prompt(
    args=(), kwargs={"prompt": "kwarg-prompt"}
)
assert _p == "kwarg-prompt", f"generate(kwarg) capture broken: p={_p!r}"

print(
    "OK: run_mlevolve.py + MLEvolve + mleval analyzer + skill_retriever "
    "+ prompt_logger all import and behave correctly"
)
