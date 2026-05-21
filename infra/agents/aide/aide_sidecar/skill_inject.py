"""If $MLEVAL_SKILL_PATH is set, splice its content into AIDE's task description.

AIDE's `aide.utils.config.load_task_desc(cfg)` reads either `cfg.desc_file`
or returns a dict built from `cfg.goal` + `cfg.eval`. The returned value is
the *task_desc* that the Agent attaches to every prompt. We monkey-patch
this loader so the skill content becomes part of the task description —
visible to every code-gen and judge call.

Activation:  `MLEVAL_SKILL_PATH` env var set to an existing readable file.
No-op:        unset / empty / non-existent file.

Why this hook (not per-prompt patching): AIDE only constructs task_desc
once at Agent.__init__ and shares it with every step's prompts. A single
splice at load time reaches every call without per-prompt monkey-patching.

Future agents may need a different hook (e.g., MLEvolve injects via an
operator-specific guideline). Each plugin owns its own injection module.
"""

from __future__ import annotations

import os
from pathlib import Path

import aide.utils.config as _config

_SKILL_PATH = os.environ.get("MLEVAL_SKILL_PATH", "").strip()
_original_load_task_desc = _config.load_task_desc


def _load_task_desc_with_skill(cfg):
    desc = _original_load_task_desc(cfg)
    if not _SKILL_PATH:
        return desc
    p = Path(_SKILL_PATH)
    if not p.is_file():
        # Silent no-op: pilot orchestrator can set MLEVAL_SKILL_PATH to a
        # path that may not yet be staged; better to run baseline than crash.
        return desc

    skill_md = p.read_text()
    if isinstance(desc, dict):
        # Don't mutate caller-owned dict; clone and add a new section.
        new = dict(desc)
        new["Available skill"] = skill_md
        return new

    # String desc: append after a horizontal rule.
    return f"{desc}\n\n---\n# Available skill\n\n{skill_md}\n"


_config.load_task_desc = _load_task_desc_with_skill
