"""Replace MLEvolve's hardcoded "all packages are installed!" lie with the
actual pip-frozen versions of libraries the agent is likely to touch.

Spike-009 verdict: even with the skill, retrieval, fence-priming fix, and Pro
respecting the response-format directive, the trajectory burned 64 min and
exited with 5/5 buggy nodes — all from ImportError chains. The agent generated
``from trl import DataCollatorForCompletionOnlyLM`` (a class that was renamed
in newer trl versions) because the prompt told it "all packages are installed!"
without naming versions. trl 1.5.1 doesn't have that class.

Survey of MLE-Bench / AIDE / MLEvolve / OpenHands (delegated research, on file)
confirmed: zero production ML agent frameworks inject pinned pip-freeze data.
The literature shows agents reach 75% version-specification rate when shown a
pinned list (vs <5% baseline). This file closes that gap.

What this patches:
  ``agents.prompts.environment.get_prompt_environment`` — the function whose
  result is dict-merged into every draft prompt at ``draft_agent.py:154``.

Dual-bind invariant (same pattern as prompt_overlay):
  ``agents/prompts/__init__.py:9`` does ``from .environment import
  get_prompt_environment`` and re-exports it at the package level. ``draft_agent``
  does ``from agents.prompts import get_prompt_environment``, which resolves
  through the package-level binding. Patching ONLY the defining submodule
  leaves callers pointing at the unpatched original. We rebind both.

Source of the pinned list:
  Read from ``$MLEVAL_INSTALLED_PACKAGES_PATH`` at import time (default
  ``/opt/agent/installed_packages.txt``). The entrypoint dumps a filtered
  ``pip freeze`` there before the agent starts, so the list reflects the
  actual container image — not whatever was true when this file was written.

Fall-back behaviour:
  If the file is missing or empty, we still patch but emit a degraded message
  that at least names the patch point ("file missing — agent has no version
  info") — better than silently re-installing the upstream lie.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import agents.prompts as _prompts_pkg
import agents.prompts.environment as _env

logger = logging.getLogger(__name__)

PATH_ENV_VAR = "MLEVAL_INSTALLED_PACKAGES_PATH"
DEFAULT_PATH = "/opt/agent/installed_packages.txt"


def _read_pinned_list() -> str:
    path = Path(os.environ.get(PATH_ENV_VAR, DEFAULT_PATH))
    if not path.is_file():
        return ""
    try:
        return path.read_text().strip()
    except OSError as e:
        logger.warning("[env_overlay] failed to read %s: %s", path, e)
        return ""


def _patched_get_prompt_environment() -> dict[str, str]:
    """Return the actual installed-package pin list under the same dict key
    MLEvolve's upstream uses, so the existing ``prompt["Instructions"] |=``
    dict-merge in ``draft_agent.py:154`` continues to splice it cleanly.
    """
    pinned = _read_pinned_list()
    if pinned:
        body = (
            "These are the EXACT package versions installed in your runtime."
            " Use the API surface of these specific versions — do not assume"
            " newer features exist (e.g. classes added in later releases will"
            " raise ImportError). If you import a name that does not exist in"
            " the version below, your run will crash:\n\n"
            f"```\n{pinned}\n```\n\n"
            "For neural networks use PyTorch. The packages above are the"
            " curated set that matters for fine-tuning, evaluation, and data"
            " loading; standard library + numpy/pandas/scikit-learn are also"
            " available but not version-pinned here."
        )
    else:
        body = (
            "(env_overlay: installed_packages.txt missing — no pinned versions"
            " available. Use only conservative APIs and test imports defensively.)"
        )
    return {"Installed Packages": body}


# ---------------------------------------------------------------------------
# Dual-bind: defining submodule + package re-export site
# ---------------------------------------------------------------------------
_env.get_prompt_environment = _patched_get_prompt_environment
_prompts_pkg.get_prompt_environment = _patched_get_prompt_environment

logger.info(
    "[env_overlay] get_prompt_environment patched (pinned list path: %s, exists: %s)",
    os.environ.get(PATH_ENV_VAR, DEFAULT_PATH),
    Path(os.environ.get(PATH_ENV_VAR, DEFAULT_PATH)).is_file(),
)
