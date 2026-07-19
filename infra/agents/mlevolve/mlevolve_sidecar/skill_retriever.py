"""Skill library loader — Anthropic progressive-disclosure (Discovery tier data).

Loads a LIBRARY of skills at import time and exposes the data the injector
(`skill_injector.py`) needs to realise progressive disclosure for MLEvolve's
single-shot codegen agents:

  - Discovery: ``catalog_text()`` — every skill's name + 1-line description +
    its ``references/*.md`` filenames. The injector splices this into EVERY
    codegen node so the agent is always aware of the whole library.
  - Activation/Execution: ``loaded_skills()`` — per-skill ``body`` (SKILL.md,
    frontmatter stripped) and ``references`` ({filename: text}). The injector's
    model selector picks which skill bodies + which references to load per node.
  - Capability linking (optional; HLD §10): if a ``capabilities.json`` manifest
    sits beside ``SKILL.md`` and validates, its units attach to the skill dict
    (``capabilities`` + ``capabilities_valid``). ``loaded_capability_skills()``
    exposes only the skills with a valid, non-empty manifest for the linker. A
    skill without a manifest is untouched and stays on the legacy path.

Source of skills (first that resolves wins):
  1. ``MLEVAL_SKILL_LIBRARY`` — a directory; scan ``*/SKILL.md`` (skip dirs
     whose name starts with ``_``). This is the library model: all skills
     always available; the selector routes.
  2. ``MLEVAL_SKILL_PATHS`` / ``MLEVAL_SKILL_PATH`` — colon-separated explicit
     SKILL.md (or skill-dir) paths. Back-compat.

This module no longer patches any MLEvolve prompt function — patching moved to
``skill_injector.py`` (the old ``get_prompt_environment`` patch reached only the
draft node; the injector reaches all four codegen stages via the universal
``get_impl_guideline_from_agent`` seam).
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

LIBRARY_ENV = "MLEVAL_SKILL_LIBRARY"
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


def _load_capabilities(skill_dir: Path, skill: dict) -> None:
    """Attach an optional ``capabilities.json`` manifest to ``skill`` in place.

    The capability-linker MVP (HLD §10, §14) ships ``capabilities.json`` BESIDE
    ``SKILL.md``. When present and valid it powers the capability delivery modes;
    when absent or invalid the skill stays on the legacy body/reference path. This
    loader is purely additive:

      - success (manifest validates): skill['capabilities'] = the unit list,
        skill['capabilities_valid'] = True.
      - failure (missing/unreadable/invalid): skill['capabilities'] stays [],
        skill['capabilities_valid'] stays False, skill['capabilities_error'] = the
        first validation error, and a warning is logged.

    A broken or absent manifest MUST NOT fail skill loading — the legacy body still
    loads (HLD §10.4 / §14 backward-compat). ``capability_schema`` is imported
    lazily so this module's import stays side-effect-free w.r.t. import order.
    """
    cap_path = skill_dir / "capabilities.json"
    if not cap_path.is_file():
        return
    try:
        from . import capability_schema  # lazy: keep loader import order clean
        manifest, errors, warnings = capability_schema.load_and_validate(
            cap_path, skill_dir=skill_dir
        )
    except Exception as e:  # noqa: BLE001 — a manifest error must never break loading
        skill["capabilities_error"] = f"{type(e).__name__}: {e}"
        logger.warning(
            "[skill_retriever] %s: capabilities.json load raised: %s", skill["name"], e
        )
        return
    if manifest is not None:  # load_and_validate returns the manifest only when valid
        skill["capabilities"] = manifest.get("capabilities") or []
        skill["capabilities_valid"] = True
        for w in warnings:
            logger.info(
                "[skill_retriever] %s: capabilities warning: %s", skill["name"], w
            )
        logger.info(
            "[skill_retriever] %s: loaded %d capability unit(s)",
            skill["name"], len(skill["capabilities"]),
        )
    else:
        skill["capabilities_error"] = errors[0] if errors else "invalid manifest"
        logger.warning(
            "[skill_retriever] %s: capabilities.json invalid: %s",
            skill["name"], skill["capabilities_error"],
        )


def _load_skill(skill_dir: Path) -> dict:
    """Load one skill: SKILL.md body (frontmatter stripped) + references map.

    Returns a dict with:
      - name, description, source_dir
      - body            — SKILL.md body only (NOT references — those load on demand)
      - references      — {filename: text} for each references/*.md
      - reference_files — sorted list of reference filenames (for the catalog)
      - capabilities        — validated capability units from capabilities.json, or []
      - capabilities_valid  — True iff a valid manifest loaded (default False)
      - capabilities_error  — first validation error when a manifest failed (optional)
    """
    skill_md = (skill_dir / "SKILL.md").read_text()
    name, description, body = _parse_frontmatter(skill_md)
    if not name:
        name = skill_dir.name

    references: dict[str, str] = {}
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        for ref_file in sorted(ref_dir.glob("*.md")):
            references[ref_file.name] = ref_file.read_text()

    skill = {
        "name": name,
        "description": description,
        "body": body,
        "references": references,
        "reference_files": sorted(references.keys()),
        "source_dir": str(skill_dir),
        # Capability-linker MVP fields (default: legacy path, no manifest).
        "capabilities": [],
        "capabilities_valid": False,
    }
    _load_capabilities(skill_dir, skill)
    return skill


def _load_library_dir(root: str) -> list[dict]:
    """Scan a library directory: every ``*/SKILL.md`` (skip ``_``-prefixed dirs)."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        logger.warning("[skill_retriever] library dir not found: %s", root)
        return []
    skills: list[dict] = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if not (child / "SKILL.md").is_file():
            continue
        try:
            skills.append(_load_skill(child))
            logger.info("[skill_retriever] loaded library skill: %s", child.name)
        except OSError as e:
            logger.warning("[skill_retriever] failed to load %s: %s", child, e)
    return skills


def _load_explicit_paths(paths_str: str) -> list[dict]:
    """Load from a colon-separated list of SKILL.md / skill-dir paths (back-compat)."""
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


def _load_all_skills() -> list[dict]:
    """Prefer MLEVAL_SKILL_LIBRARY (dir scan); else MLEVAL_SKILL_PATHS/PATH."""
    library = os.environ.get(LIBRARY_ENV, "").strip()
    if library:
        return _load_library_dir(library)
    paths_str = os.environ.get(PATH_ENV_MULTI) or os.environ.get(PATH_ENV_SINGULAR, "")
    if paths_str.strip():
        return _load_explicit_paths(paths_str)
    return []


_SKILLS: list[dict] = _load_all_skills()


def catalog_text() -> str:
    """L1 catalog (Discovery): name + 1-line description + reference filenames."""
    lines = []
    for s in _SKILLS:
        desc = (s["description"] or "(no description)").strip()
        lines.append(f"- **{s['name']}**: {desc}")
        if s["reference_files"]:
            lines.append(f"    references: {', '.join(s['reference_files'])}")
    return "\n".join(lines)


logger.info("[skill_retriever] loaded %d skill(s) into library", len(_SKILLS))


# ---------------------------------------------------------------------------
# Test/smoke helpers (used by _smoke_imports.py and reload-during-test paths)
# ---------------------------------------------------------------------------

def loaded_skills() -> list[dict]:
    """Return the current loaded-skill list (post-import; may be empty)."""
    return list(_SKILLS)


def loaded_capability_skills() -> list[dict]:
    """Return only skills carrying a VALID, non-empty capabilities manifest.

    These are the ``cap_skills`` the ``capability_linker`` consumes (HLD §10). A
    skill without ``capabilities.json``, or with an invalid or empty one, is
    excluded: capability delivery modes must NOT mix in that skill's legacy body
    (HLD §10.4 — a capability arm never silently emits legacy content). An empty
    result is the no-skill baseline for the capability modes.
    """
    return [
        s for s in _SKILLS
        if s.get("capabilities_valid") and s.get("capabilities")
    ]


def reload() -> int:
    """Re-read the skill env vars and return the new skill count. Test-only."""
    global _SKILLS
    _SKILLS = _load_all_skills()
    return len(_SKILLS)
