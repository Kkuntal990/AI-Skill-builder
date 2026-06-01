"""Replace MLEvolve's stock package-hint list with a contract-aligned one.

MLEvolve's upstream ``get_prompt_environment`` (``agents/prompts/environment.py``)
ships a hand-curated 15-package list biased toward Kaggle tabular/CV work —
xgboost, lightGBM, timm, opencv-python, Pillow — none of which are relevant to
text generation tasks. That list flows into every draft prompt.

Earlier iterations of this file replaced it with the full filtered pip-freeze
output (peft==0.12.0, trl==…, bitsandbytes==…). That worked to prevent
spike-009 ImportError cascades but BIASED the A/B: it advertised peft/trl
specifically to both cells, signalling the skill's recommended libraries to
the without_skill control.

Current approach (matches upstream MLEvolve's style as published in the
AutoMLGen MLE-Bench eval):

  - hand-curated, contract-aligned hint list (10 packages — generic ML/LLM
    stack: no peft/trl/bitsandbytes, no Kaggle-tabular/CV libs)
  - no version numbers (rely on requirements.txt pinning to mid-2024
    versions that match the LLM's training-data prior)
  - explicit "all packages installed" disclaimer + discovery hint so the
    agent knows it can import anything and verify with ``pip show``

Spike-009 protection now comes from requirements.txt pinning, not from
listing versions in the prompt. See infra/agents/mlevolve/requirements.txt.

Dual-bind invariant (same pattern as prompt_overlay):
  ``agents/prompts/__init__.py:9`` does ``from .environment import
  get_prompt_environment`` and re-exports it at the package level.
  ``draft_agent`` does ``from agents.prompts import get_prompt_environment``,
  which resolves through the package-level binding. Patching ONLY the
  defining submodule leaves callers pointing at the unpatched original.
  We rebind both.
"""
from __future__ import annotations

import logging

import agents.prompts as _prompts_pkg
import agents.prompts.environment as _env

logger = logging.getLogger(__name__)

# Contract-aligned package hint list: stable HF stack libraries that any
# fine-tuning / evaluation / data-loading approach would touch. Deliberately
# excludes peft/trl/bitsandbytes (those are method-bias for our PEFT A/B —
# the skill's recommendation, not generic ML hygiene). Also excludes
# xgboost/lightGBM/timm/opencv-python from upstream's list (Kaggle-only,
# noise for text tasks). Deterministic order — no random.shuffle — so paired
# seeds across cells see identical prompts.
_PKG_HINTS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
    "datasets",
    "evaluate",
    "accelerate",
    "tokenizers",
    "safetensors",
)


def _patched_get_prompt_environment() -> dict[str, str]:
    """Return a contract-aligned package hint under MLEvolve's expected key.

    Shape matches upstream so ``prompt["Instructions"] |= result`` in
    ``draft_agent.py:154`` continues to splice cleanly.
    """
    pkg_str = ", ".join(f"`{p}`" for p in _PKG_HINTS)
    body = (
        f"Your solution can use any relevant machine learning packages such as: "
        f"{pkg_str}. Feel free to use any other packages too (all packages are "
        f"already installed!). For neural networks use PyTorch rather than "
        f"TensorFlow. If you are uncertain whether a specific library is "
        f"available or what version is installed, you can verify at runtime by "
        f"importing it and printing its `__version__`, or by inspecting the "
        f"output of `pip show <package>`."
    )
    return {"Installed Packages": body}


# ---------------------------------------------------------------------------
# Dual-bind: defining submodule + package re-export site
# ---------------------------------------------------------------------------
_env.get_prompt_environment = _patched_get_prompt_environment
_prompts_pkg.get_prompt_environment = _patched_get_prompt_environment

logger.info(
    "[env_overlay] get_prompt_environment patched (contract-aligned hint, %d pkgs, no versions)",
    len(_PKG_HINTS),
)
