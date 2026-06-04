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
# skill injection guards (Anthropic progressive disclosure — library + per-node
# model selector). Catches: the per-agent rebind failing (re-export
# propagation), the _-prefixed-dir skip regressing, frontmatter parsing drift,
# the selector FunctionSpec going malformed, and the empty-library baseline
# leaking a catalog.
# -----------------------------------------------------------------------------
import tempfile as _tf0  # noqa: E402
import pathlib as _pl  # noqa: E402

import agents.draft_agent as _draft  # noqa: E402
import agents.improve_agent as _improve  # noqa: E402
import agents.debug_agent as _debug  # noqa: E402
import agents.evolution_agent as _evolution  # noqa: E402
from mlevolve_sidecar import skill_retriever  # noqa: E402
from mlevolve_sidecar import skill_injector  # noqa: E402

# 1. Per-agent rebind fired for ALL FOUR codegen agents (the core fix vs the
#    old draft-only gap). Patching the definition / package re-export does NOT
#    change these module bindings — so this is the load-bearing assertion.
for _mod in (_draft, _improve, _debug, _evolution):
    assert getattr(_mod, "_mleval_skill_patched", False), \
        f"skill_injector did not patch {_mod.__name__} (import hook missed it?)"
    assert getattr(_mod.run, "_mleval_patched", False), \
        f"{_mod.__name__}.run is not the skill-wrapped run"
    assert getattr(_mod.get_impl_guideline_from_agent, "_mleval_patched", False), \
        f"{_mod.__name__}.get_impl_guideline_from_agent is not the skill-wrapped fn"

# 2. Selector FunctionSpec builds and renders to an OpenAI tool dict (catches a
#    malformed json_schema at build time, before a cluster run).
_spec = skill_injector._get_selector_spec()
assert _spec.name == "select_skills", f"selector spec name changed: {_spec.name}"
assert _spec.as_openai_tool_dict["function"]["name"] == "select_skills", \
    "selector FunctionSpec.as_openai_tool_dict malformed"

# 3. Empty-library baseline — with no skills loaded, the wrapped guideline fn
#    must return the upstream list UNCHANGED (no catalog, no selector call).
os.environ.pop("MLEVAL_SKILL_LIBRARY", None)
os.environ.pop("MLEVAL_SKILL_PATHS", None)
os.environ.pop("MLEVAL_SKILL_PATH", None)
assert skill_retriever.reload() == 0, "expected 0 skills with no env var"
assert skill_retriever.catalog_text() == "", "catalog should be empty with no skills"
_baseline = {"Implementation guideline": ["original-line"]}
_wrapped = skill_injector._wrap_impl_guideline(lambda _agent: _baseline)
_out = _wrapped(object())  # stub agent; must not be touched when no skills
assert _out["Implementation guideline"] == ["original-line"], \
    f"empty-library path mutated the guideline: {_out}"

# 4. Library round-trip — a synthetic library with a real skill dir and a
#    _-prefixed dir that MUST be skipped; catalog lists the skill + its refs.
with _tf0.TemporaryDirectory() as _libdir:
    _lib = _pl.Path(_libdir)
    _sk = _lib / "demo-skill"
    (_sk / "references").mkdir(parents=True)
    (_sk / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: a demo skill for smoke\n---\n\nBody here.\n"
    )
    (_sk / "references" / "guide.md").write_text("# Guide\nref content\n")
    _hidden = _lib / "_hidden"
    _hidden.mkdir()
    (_hidden / "SKILL.md").write_text("---\nname: hidden\ndescription: skip me\n---\nx\n")

    os.environ["MLEVAL_SKILL_LIBRARY"] = str(_lib)
    _n = skill_retriever.reload()
    assert _n == 1, f"library scan loaded {_n} skills, expected 1 (_hidden skipped)"
    _loaded = skill_retriever.loaded_skills()
    assert _loaded[0]["name"] == "demo-skill", f"unexpected load: {_loaded}"
    assert _loaded[0]["reference_files"] == ["guide.md"], \
        f"reference_files wrong: {_loaded[0]['reference_files']}"
    assert "guide.md" in _loaded[0]["references"], "reference body not loaded"
    _cat = skill_retriever.catalog_text()
    assert "demo-skill" in _cat and "guide.md" in _cat, f"catalog missing entries: {_cat}"
    assert "hidden" not in _cat, "_-prefixed dir leaked into catalog"

    os.environ.pop("MLEVAL_SKILL_LIBRARY", None)
    assert skill_retriever.reload() == 0, "reset to 0 skills failed"

# -----------------------------------------------------------------------------
# token_budget guard — the anti-truncation sidecar must have wrapped the
# provider-level query/generate so the default max_tokens is raised (spike-012
# corruption root cause was 16384-token truncation). Catches: import-order
# regression, or upstream renaming llm.openai.query/generate.
# -----------------------------------------------------------------------------
import llm.openai as _openai_provider  # noqa: E402

assert getattr(_openai_provider.query, "_token_budget_patched", False), \
    "token_budget did not wrap llm.openai.query (max_tokens cap not raised)"
assert getattr(_openai_provider.generate, "_token_budget_patched", False), \
    "token_budget did not wrap llm.openai.generate (max_tokens cap not raised)"

# -----------------------------------------------------------------------------
# bitsandbytes import guard — the base image ships triton 3.3.1, which dropped
# `triton.ops`. bnb 0.43.3 eagerly imported `triton.ops.matmul_perf_model` and
# died at import, breaking every QLoRA/4-bit path (the spike-012 confound).
# 0.46.1 fixes this; assert it imports AND that the 4-bit config the skill
# recommends constructs (CPU-only construction, no CUDA needed at build time).
# -----------------------------------------------------------------------------
import bitsandbytes as _bnb  # noqa: F401,E402 — must import without triton.ops
from transformers import BitsAndBytesConfig as _BnbCfg  # noqa: E402

_bnb_cfg = _BnbCfg(load_in_4bit=True, bnb_4bit_quant_type="nf4")
assert _bnb_cfg.load_in_4bit, "BitsAndBytesConfig(load_in_4bit=True) did not stick"

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

# -----------------------------------------------------------------------------
# held-out grader guard — the trustworthy A/B metric path. The grader must
# import and deterministically (a) score a perfect match as 1.0 and (b) REJECT
# an id-set mismatch (the drift signature) rather than silently scoring it.
# -----------------------------------------------------------------------------
from mleval.grader import grade_predictions as _grade  # noqa: E402
from mleval.grader import rouge_l_f as _rouge  # noqa: E402

assert abs(_rouge("the cat sat", "the cat sat") - 1.0) < 1e-9, "rouge_l_f identity != 1.0"
assert _rouge("alpha beta", "gamma delta") == 0.0, "rouge_l_f disjoint != 0.0"

import csv as _csv  # noqa: E402
import pathlib as _pl2  # noqa: E402
import tempfile as _tf  # noqa: E402


def _write_csv2(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


with _tf.TemporaryDirectory() as _d:
    _dp = _pl2.Path(_d)
    _rows = [["1", "the cat sat"], ["2", "a dog ran"]]
    _write_csv2(_dp / "refs.csv", ["id", "reference_summary"], _rows)
    _write_csv2(_dp / "preds.csv", ["id", "generated_summary"], _rows)
    _gr = _grade(_dp / "preds.csv", _dp / "refs.csv")
    assert _gr.valid and abs(_gr.score - 1.0) < 1e-9, f"grader perfect-match failed: {_gr}"
    _write_csv2(_dp / "drift.csv", ["id", "generated_summary"], [["0", "positive"], ["1", "neg"]])
    _gd = _grade(_dp / "drift.csv", _dp / "refs.csv")
    assert _gd.valid is False and _gd.score is None, f"grader failed to reject drift: {_gd}"

# -----------------------------------------------------------------------------
# de_kaggle split-safety guard — with no_submission_mode:False the result
# parser runs _validate_format_with_retry, whose `exp_name.split("_")[2]` must
# be the index-safe form the de_kaggle build patch installs (else IndexError on
# our hyphenated exp_name, before the use_grading_server=False skip).
# -----------------------------------------------------------------------------
import inspect as _inspect  # noqa: E402

import agents.result_parse_agent as _rpa  # noqa: E402

_rpa_src = _inspect.getsource(_rpa)
assert 'exp_name.split("_")[2]' not in _rpa_src, \
    'de_kaggle split-safety patch did not apply: raw exp_name.split("_")[2] still present'
assert 'exp_name.split("_") + ["", "", ""]' in _rpa_src, \
    "de_kaggle split-safety patch missing the index-safe form"

print(
    "OK: run_mlevolve.py + MLEvolve + mleval analyzer + skill_retriever "
    "+ prompt_logger + grader + de_kaggle split-safety all import and behave correctly"
)
