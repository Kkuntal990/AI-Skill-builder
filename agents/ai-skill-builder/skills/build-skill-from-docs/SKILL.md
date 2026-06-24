---
name: build-skill-from-docs
description: "Generate an OpenClaw SKILL.md from a Python package's documentation URL. Fetches doc + README + examples (and optionally GitHub issues), synthesizes a progressive-disclosure skill (SKILL.md + references + evals), and installs safely. Use when the user says 'build a skill for X', 'turn this doc into a skill', or provides a package doc URL. For describing a skill interactively rather than building one from docs, use anthropic's skill-creator; to discover skills that already exist, use find-ai-skill."
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

A closed generate -> gate -> critic -> repair loop (not a one-shot template):

```
RESOLVE -> FETCH (doc, README, examples, [issues], [changelog])
       -> PLAN STRUCTURE (LLM: references, decision tree, MCP triggers, scripts)
       -> WRITE BODY (LLM)
       -> CRITIC + REPAIR (<=3 rounds: critique P1-P4 -> repair body if blocking)
       -> SYNTHESIZE in parallel (references + templates + scripts + evals)
       -> TRIGGERING eval (judge vs siblings/decoys; optimize description on a miss)
       -> ASSEMBLE frontmatter (deterministic, no LLM)
       -> VALIDATE (P0 gates + dead-pointer + security scan + line cap + openclaw check)
       -> WRITE to ~/.openclaw/workspace/skills/<name>/
```

## Best Practices Enforced

See `references/skill-anatomy.md` for progressive-disclosure rules.
See `references/frontmatter-spec.md` for OpenClaw metadata schema.
See `references/anti-patterns.md` for traps the script rejects automatically.

Two gates run before a skill is written:

**P0 — deterministic validator (hard-fail; rejects the build):**

- SKILL.md body <= 500 lines (target 30-80)
- Frontmatter parses as YAML; `name` matches `^[a-z0-9-]{1,64}$`, is not a reserved word (anthropic/claude), and is XML-free
- `description` is <= 1024 chars and XML-free, third person, stating both what the skill does and when to use it (a "Use when..." clause is expected, not forbidden)
- No dead pointers: every `references/`, `scripts/`, `templates/` path cited in the body must be bundled
- No BLOCK-pattern shell commands; no commands absent from the source docs (catches fabrication)

**P1-P4 — quality critic + bounded repair (<=3 rounds; ship-with-warning):**

- **Scope-honesty (P4)** — flags task-genre "NOT for" walls (e.g. an inference skill excluding fine-tuning); these are auto-repaired before shipping
- **Description (P1)**, **content anti-patterns (P3:** ALL-CAPS directives, Windows paths, time-sensitive language**)**, **focus (P2:** >3 reference modules**)**
- Residual findings ship as warnings; the build report's `quality_gate` field carries pass/fail. Deterministic checks are regex; only P4 + semantic P1 use one abstain-when-unsure LLM call.

## LLM

Synthesis runs through a single `_llm_call` dispatcher, selected by
`MLEVAL_LLM_TRANSPORT`:

- `claude` (**default**) -- the local Claude Code CLI (`claude -p`), which uses your
  Claude **subscription** (no per-token API credit) and Claude models (the CLI's
  default, typically Opus; override with `MLEVAL_LLM_MODEL`). Falls back to OpenRouter
  if the `claude` binary is missing/fails and an OpenRouter key is configured.
- `openrouter` -- direct OpenRouter API (**paid credit**). Key from `OPENROUTER_API_KEY`
  or the OpenClaw `openrouter:default` auth profile; model = `OPENROUTER_MODEL`
  (`anthropic/claude-sonnet-4.6`).

Script exits clearly if no transport is available.

## Optional flags

- `--with-scripts` -- generate 1-3 SHORT utility scripts in `scripts/` (executed, not
  read into context). Validated via `py_compile` / `bash -n`. Use for health checks,
  validators, probes the agent should run unattended.
- `--with-version-notes` -- emit an `## Old Patterns` collapsed section listing deprecated APIs.
- `--with-pitfalls` / `--with-troubleshooting` / `--with-community` -- distill GitHub
  issues / Stack Exchange Q&As into extra reference files.
- `--siblings <dir>` -- run the triggering eval against the REAL co-resident skills in
  `<dir>` (each `<dir>/<name>/SKILL.md`) instead of the canned decoys; also enables
  false-positive (over-trigger) detection.
- `--no-critic` -- skip the quality critic + repair loop (P1-P4).

The planner emits `decision_tree` (routing table for choices between modes) and
per-workflow `mcp_workflow_triggers` when the library warrants them -- default-on, no flag.

## Output

The script prints a JSON summary to stdout with the output path, files written,
validation warnings, and source URLs. Present results conversationally.
