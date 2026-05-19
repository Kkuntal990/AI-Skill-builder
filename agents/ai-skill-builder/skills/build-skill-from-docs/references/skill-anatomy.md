# Skill Anatomy — Progressive Disclosure

Source: adapted from `anthropics/skills/skill-creator`.

## Three-Level Loading

1. **Metadata** (name + description) — Always in context (~100 words). This is the primary trigger.
2. **SKILL.md body** — In context whenever the skill triggers. Target 30-80 lines, hard cap 500.
3. **Bundled resources** — Loaded on demand:
   - `scripts/` — Executable code for deterministic work
   - `references/` — Docs loaded into context when Claude reads them
   - `assets/` — Output templates, fonts, icons
   - `evals/` — Test prompts (not auto-loaded)

## Layout

```text
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter (name, description required)
│   └── Markdown instructions (short)
├── references/           (heavy docs, domain-variant splits)
├── scripts/              (deterministic operations)
├── assets/               (output templates)
└── evals/evals.json      (2-3 realistic test prompts)
```

## When to Split into references/

- Content exceeds 20 lines AND is not always needed
- Domain variants (e.g. `cloud-deploy/references/{aws,gcp,azure}.md`)
- Large code examples (full training loops, not snippets)
- API reference tables

Keep in SKILL.md:

- Quick start (the fastest path from zero to "it worked")
- Pointers to reference files ("For DPO specifics, see `references/dpo.md`")
- The imperative core rules

## Domain Variant Grouping

Don't produce one reference file per variant when there are many. Group by method class:

- TRL has 12+ trainers → group into 3-4 files (offline methods, online methods, reward modeling, knowledge distillation)
- Transformers has hundreds of models → one reference for each model *family*, not each checkpoint

The `plan` step decides this.
