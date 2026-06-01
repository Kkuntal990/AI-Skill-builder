"""Skill injection — Anthropic two-tier (catalog + full body), no retrieval.

Reads ``MLEVAL_SKILL_PATHS`` (colon-separated list, falls back to
``MLEVAL_SKILL_PATH`` singular for backward compat) at import time. For each
path:
  - Resolves a SKILL.md file path OR a directory containing SKILL.md
  - Loads SKILL.md content (full body, after stripping YAML frontmatter)
  - Concatenates any ``references/*.md`` files (full)
  - Parses frontmatter for L1 catalog metadata (name + description)

Patches ``agents.prompts.environment.get_prompt_environment`` to splice the
catalog and full skill bodies into the dict that ``draft_agent.py:154`` merges
into ``prompt["Instructions"]``. Stepwise's StepAgent / MetaAgent
(``coder/stepwise_coder.py``) copy ``prompt_base["Instructions"]`` into their
own prompts, so the skill content reaches the live codegen path even though
stepwise bypasses ``build_chat_prompt_for_model``.

Slot layout in ``prompt["Instructions"]`` (only when at least one skill loads):
  - ``"Available Skills"`` — L1 catalog (skill name + 1-line description)
  - ``"Skill Reference"`` — L2 full content (SKILL.md body + all references)

Design rationale (see conversation 2026-06-01):
  - Matches Anthropic's published Skills API: caller pre-selects skills (we
    pass them via the env var); progressive disclosure shape with metadata +
    body; up-to-8-skills cap matches Anthropic's API limit.
  - Replaces ~400 LoC BM25 retriever. No chunker, no embedding, no stage
    detection, no two-gate threshold. Trust the skill author's references/
    structure and the LLM's ability to skim.
  - Bypassed-by-stepwise problem (spike-011): the OLD implementation injected
    into ``build_chat_prompt_for_model`` which stepwise's coder never calls.
    The ``Instructions`` dict slot is the one place that survives all of:
    non-stepwise draft, stepwise StepAgent + MetaAgent, improve, debug.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import agents.prompts as _prompts_pkg
import agents.prompts.environment as _env_mod

logger = logging.getLogger(__name__)

PATH_ENV_MULTI = "MLEVAL_SKILL_PATHS"
PATH_ENV_SINGULAR = "MLEVAL_SKILL_PATH"

# YAML frontmatter parser — minimal (we only need name + description).
# Avoids adding pyyaml as a runtime dep. SKILL.md frontmatter follows the
# Anthropic skill schema: ``---\nname: foo\ndescription: bar\n---\n<body>``.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


def _resolve_skill_dir(p: str) -> Path | None:
    """Accept either a SKILL.md file path OR a skill directory path."""
    path = Path(p).resolve()
    if path.is_file() and path.name == "SKILL.md":
        return path.parent
    if path.is_dir() and (path / "SKILL.md").is_file():
        return path
    return None


def _parse_frontmatter(skill_md: str) -> tuple[str, str, str]:
    """Return (name, description, body). Falls back to dir name + empty desc."""
    m = _FRONTMATTER_RE.match(skill_md)
    if not m:
        return "", "", skill_md
    fm_text, body = m.group(1), m.group(2)
    name = ""
    description = ""
    for raw in fm_text.splitlines():
        line = raw.rstrip()
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("\"'")
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("\"'")
    return name, description, body


def _load_skill(skill_dir: Path) -> dict:
    """Load one skill: SKILL.md + concatenated references/*.md."""
    skill_md = (skill_dir / "SKILL.md").read_text()
    name, description, body = _parse_frontmatter(skill_md)
    if not name:
        name = skill_dir.name

    ref_blocks: list[str] = []
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        for ref_file in sorted(ref_dir.glob("*.md")):
            ref_blocks.append(
                f"\n\n#### references/{ref_file.name}\n\n{ref_file.read_text()}"
            )

    return {
        "name": name,
        "description": description,
        "body": body + "".join(ref_blocks),
        "source_dir": str(skill_dir),
    }


def _load_all_skills() -> list[dict]:
    """Read MLEVAL_SKILL_PATHS (preferred) or MLEVAL_SKILL_PATH (back-compat)."""
    paths_str = os.environ.get(PATH_ENV_MULTI) or os.environ.get(PATH_ENV_SINGULAR, "")
    if not paths_str.strip():
        return []
    skills: list[dict] = []
    for raw in paths_str.split(":"):
        path = raw.strip()
        if not path:
            continue
        skill_dir = _resolve_skill_dir(path)
        if skill_dir is None:
            logger.warning("[skill_retriever] skip unresolvable path: %s", path)
            continue
        try:
            skills.append(_load_skill(skill_dir))
            logger.info("[skill_retriever] loaded skill: %s", skill_dir.name)
        except OSError as e:
            logger.warning("[skill_retriever] failed to load %s: %s", skill_dir, e)
    return skills


_SKILLS: list[dict] = _load_all_skills()


def _catalog_text() -> str:
    """L1: skill names + descriptions, formatted for the Instructions dict."""
    lines = []
    for s in _SKILLS:
        desc = (s["description"] or "(no description)").strip()
        lines.append(f"- **{s['name']}**: {desc}")
    return "\n".join(lines)


def _reference_text() -> str:
    """L2: full SKILL.md body + references for every loaded skill."""
    blocks = []
    for s in _SKILLS:
        blocks.append(f"### Skill: {s['name']}\n\n{s['body']}")
    return "\n\n---\n\n".join(blocks)


_orig_get_prompt_environment = _env_mod.get_prompt_environment


def _patched_get_prompt_environment() -> dict:
    """Upstream env dict + skill catalog + skill body (when skills are loaded)."""
    result = dict(_orig_get_prompt_environment())
    if _SKILLS:
        result["Available Skills"] = _catalog_text()
        result["Skill Reference"] = _reference_text()
    return result


# Dual-bind: defining submodule + package re-export site (mirrors the pattern
# the deleted env_overlay used; agents import via the re-export, so patching
# only the defining submodule leaves callers pointing at the original).
_env_mod.get_prompt_environment = _patched_get_prompt_environment
_prompts_pkg.get_prompt_environment = _patched_get_prompt_environment

logger.info(
    "[skill_retriever] loaded %d skill(s); patched get_prompt_environment",
    len(_SKILLS),
)


# ---------------------------------------------------------------------------
# Test/smoke helpers (used by _smoke_imports.py and reload-during-test paths)
# ---------------------------------------------------------------------------

def loaded_skills() -> list[dict]:
    """Return the current loaded-skill list (post-import; may be empty)."""
    return list(_SKILLS)


def reload() -> int:
    """Re-read MLEVAL_SKILL_PATH(S) and return the new skill count. Test-only."""
    global _SKILLS
    _SKILLS = _load_all_skills()
    return len(_SKILLS)
