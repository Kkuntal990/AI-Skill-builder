"""If $MLEVAL_SKILL_PATH is set, splice the skill into AIDE's task description.

AIDE's `aide.utils.config.load_task_desc(cfg)` reads either `cfg.desc_file`
or returns a dict built from `cfg.goal` + `cfg.eval`. The returned value is
the *task_desc* that the Agent attaches to every prompt. We monkey-patch
this loader so the skill content becomes part of the task description —
visible to every code-gen and judge call.

Activation:  `MLEVAL_SKILL_PATH` env var set to an existing readable path.
             May point to either:
               * a single SKILL.md file — if a sibling `references/`
                 directory exists, its `*.md` files are auto-concatenated
                 (deterministic sort by filename so prompt ordering is
                 reproducible across runs);
               * a skill DIRECTORY containing SKILL.md plus optional
                 `references/*.md` (same concatenation behaviour).
No-op:        unset / empty / non-existent path.

Why references concatenation: OpenClaw skills use progressive disclosure —
SKILL.md is the entry-point and references/*.md are deeper material the
entry-point points to with `See \`references/X.md\`` lines. AIDE cannot
navigate the filesystem from prompts, so dangling references are dead
pointers unless we splice the whole bundle.

The concatenation prepends a "## references/<name>.md" header per file so
AIDE can see the boundaries (in case it tries to reproduce the structure
in its own outputs).

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


def get_skill_dir() -> Path | None:
    """Return the skill bundle's root directory, or None if unset/missing.

    Public helper shared with :mod:`interpreter_patch` so the scripts/ copy
    resolves the same way the references/ concat does. The skill root is:
      * ``Path(MLEVAL_SKILL_PATH).parent``  if the env var points at a file
        (the canonical case — orchestrator passes the SKILL.md path)
      * ``Path(MLEVAL_SKILL_PATH)``         if it points at a directory
      * ``None`` otherwise (unset, empty, missing path)
    """
    if not _SKILL_PATH:
        return None
    p = Path(_SKILL_PATH)
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    return None


def _concat_references(refs_dir: Path, parts: list[str]) -> None:
    if not refs_dir.is_dir():
        return
    for ref in sorted(refs_dir.glob("*.md")):
        parts.append(f"\n\n## references/{ref.name}\n\n{ref.read_text()}")


def _load_skill_content(path: Path) -> str | None:
    """Read a skill from a file or a directory; return concatenated markdown."""
    parts: list[str] = []
    if path.is_file():
        parts.append(path.read_text())
        _concat_references(path.parent / "references", parts)
    elif path.is_dir():
        skill_md = path / "SKILL.md"
        if skill_md.is_file():
            parts.append(skill_md.read_text())
        _concat_references(path / "references", parts)
    return "\n".join(parts) if parts else None


def _load_task_desc_with_skill(cfg):
    desc = _original_load_task_desc(cfg)
    if not _SKILL_PATH:
        return desc
    skill_md = _load_skill_content(Path(_SKILL_PATH))
    if skill_md is None:
        # Silent no-op: pilot orchestrator may set MLEVAL_SKILL_PATH to a
        # path that may not yet be staged; better to run baseline than crash.
        return desc

    if isinstance(desc, dict):
        # Don't mutate caller-owned dict; clone and add a new section.
        new = dict(desc)
        new["Available skill"] = skill_md
        return new

    # String desc: append after a horizontal rule.
    return f"{desc}\n\n---\n# Available skill\n\n{skill_md}\n"


_config.load_task_desc = _load_task_desc_with_skill
