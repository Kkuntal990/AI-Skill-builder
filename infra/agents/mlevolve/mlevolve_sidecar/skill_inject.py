"""Standalone helper: splice a skill bundle into MLEvolve's description.md.

Called by entrypoint.sh BEFORE the agent starts (not at runtime), because
MLEvolve's load_task_desc reads desc_file as a single blob with no per-call
monkey-patch hook. Mirrors the AIDE skill_inject content rules:

- If MLEVAL_SKILL_PATH points at a file, concat sibling references/*.md
  in deterministic sort order.
- If it points at a directory, read SKILL.md + references/*.md the same
  way.
- Prepend the spliced content under a clear marker so the analyzer can
  detect it later.

Idempotent: if description.md already contains the splice marker, do
nothing. Lets the entrypoint be re-run during debugging without
double-injecting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MARKER = "<!-- mleval-skill-injected -->"


def _concat_references(refs_dir: Path) -> list[str]:
    if not refs_dir.is_dir():
        return []
    parts: list[str] = []
    for ref in sorted(refs_dir.glob("*.md")):
        parts.append(f"\n\n## references/{ref.name}\n\n{ref.read_text()}")
    return parts


def _load_skill_content(path: Path) -> str | None:
    parts: list[str] = []
    if path.is_file():
        parts.append(path.read_text())
        parts.extend(_concat_references(path.parent / "references"))
    elif path.is_dir():
        skill_md = path / "SKILL.md"
        if skill_md.is_file():
            parts.append(skill_md.read_text())
        parts.extend(_concat_references(path / "references"))
    return "\n".join(parts) if parts else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-path", required=True)
    ap.add_argument("--description", required=True)
    args = ap.parse_args()

    skill_path = Path(args.skill_path)
    desc_path = Path(args.description)
    if not skill_path.exists():
        print(f"[skill_inject] skill not found at {skill_path}; no-op", file=sys.stderr)
        return 0
    if not desc_path.exists():
        print(f"[skill_inject] description not found at {desc_path}", file=sys.stderr)
        return 1

    current = desc_path.read_text()
    if _MARKER in current:
        print(f"[skill_inject] already injected; no-op", file=sys.stderr)
        return 0

    skill_md = _load_skill_content(skill_path)
    if skill_md is None:
        print(f"[skill_inject] skill content empty; no-op", file=sys.stderr)
        return 0

    new = f"{_MARKER}\n\n# Available skill\n\n{skill_md}\n\n---\n\n{current}"
    desc_path.write_text(new)
    print(f"[skill_inject] spliced {len(skill_md)} chars from {skill_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
