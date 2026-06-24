---
name: build-skill-from-docs
description: "Generate an OpenClaw SKILL.md from a Python package's documentation URL. Fetches doc + README + examples (and optionally GitHub issues), synthesizes a progressive-disclosure skill (SKILL.md + references + evals), and installs safely. Use when: user says 'build a skill for X', 'turn this doc into a skill', or provides a package doc URL. NOT for: describing a skill interactively (use anthropic's skill-creator), finding existing skills (use find-ai-skill)."
metadata:
  {
    "openclaw":
      {
        "emoji": "🛠️",
        "requires": { "bins": ["gh", "python3"] },
        "install":
          [
            {
              "id": "brew-gh",
              "kind": "brew",
              "formula": "gh",
              "bins": ["gh"],
              "label": "Install GitHub CLI (brew)",
            },
          ],
      },
  }
---

# Build Skill from Docs

Turn a Python package's documentation into an installable OpenClaw skill.

## Usage

```bash
# Full build
python3 scripts/skill_builder.py build https://huggingface.co/docs/trl/index

# Optional extras
python3 scripts/skill_builder.py build <url> --with-pitfalls --with-version-notes
python3 scripts/skill_builder.py build <url> --name my-skill --no-evals
python3 scripts/skill_builder.py build <url> --force  # overwrite existing

# Dry-run modes
python3 scripts/skill_builder.py preview <url>   # synthesize, print to stdout
python3 scripts/skill_builder.py plan <url>      # show file-structure decision only
python3 scripts/skill_builder.py sources <owner/repo>  # show what would be fetched

# Inspection
python3 scripts/skill_builder.py built           # list generated skills
```

## Pipeline

```
RESOLVE -> FETCH (doc, README, examples, [issues], [changelog])
       -> PLAN STRUCTURE (LLM: decide references/*.md files)
       -> SYNTHESIZE in parallel (SKILL.md body + references + evals)
       -> ASSEMBLE frontmatter (deterministic, no LLM)
       -> VALIDATE (security scan + YAML parse + line cap + openclaw skills check)
       -> WRITE to ~/.openclaw/workspace/skills/<name>/
```

## Best Practices Enforced

See `references/skill-anatomy.md` for progressive-disclosure rules.
See `references/frontmatter-spec.md` for OpenClaw metadata schema.
See `references/anti-patterns.md` for traps the script rejects automatically.

Core rules enforced by the validator:

- SKILL.md body <= 500 lines (target 30-80)
- Frontmatter must parse as YAML and include `name` + `description`
- No BLOCK-pattern shell commands in generated output
- No shell commands absent from the source docs (catches LLM fabrication)
- Description uses imperative action verbs, not "use when..."

## LLM

Claude Sonnet 4.6 via OpenRouter (constant at `scripts/skill_builder.py:OPENROUTER_MODEL`).
API key from `OPENROUTER_API_KEY` env var or OpenClaw auth store (`openrouter:default`
profile). Script exits clearly if missing -- synthesis has no useful fallback.

## Optional flags (new in 1.4.0)

- `--with-scripts` -- generate 1-3 SHORT utility scripts in `scripts/` (executed, not
  read into context). Validated via `py_compile` / `bash -n`. Use for health checks,
  validators, probes that the agent should run unattended.
- `--with-version-notes` -- emit an `## Old Patterns` collapsed section listing
  deprecated APIs (when the docs mention any).

The planner also now emits `decision_tree` (routing table for choices between modes)
and `mcp_workflow_triggers` (per-workflow inline MCP fallback instructions) when the
library warrants them -- both default-on, no flag needed.

## Output

The script prints a JSON summary to stdout with the output path, files written,
validation warnings, and source URLs. Present results conversationally.
