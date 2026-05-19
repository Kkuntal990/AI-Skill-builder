# TOOLS.md -- Local Environment

## Required Binaries

- `gh` -- GitHub CLI (authenticated; used for README, issues, examples, changelog)
- `python3` -- Python 3.14+
- `openclaw` -- OpenClaw 2026.4.12+ (for `skills check` validation)

## Required API Key

- `OPENROUTER_API_KEY` -- Claude Opus synthesis via OpenRouter. Falls back to
  OpenClaw auth store (`openrouter:default` profile) if env var is unset.
  On absence the script exits with a clear error (unlike Scout, synthesis
  has no literal-query fallback; building without an LLM is not useful).

## Paths

- Skills install to: `~/.openclaw/workspace/skills/<name>/`
- State dir: `<workspace>/data/`
  - `built-skills.json` -- lockfile of generated skills (name, source URL, content hash)
  - `doc-cache.json` -- cached doc fetches (1hr TTL)
  - `skill-builder-audit.log` -- append-only log

## Shared Modules

- Security scanner: imported from sibling `ai-skill-scout` agent
  (`agents/ai-skill-scout/skills/find-ai-skill/scripts/skill_scout.py`).
  One source of truth for BLOCK_PATTERNS and CAUTION_PATTERNS.
