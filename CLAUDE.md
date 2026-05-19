# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project scope

This repo houses three loosely-coupled bodies of work, all centered on **OpenClaw skills for ML-engineering agents**:

| Area | Where | What |
|---|---|---|
| **Skill-builder** | `agents/ai-skill-builder/`, `docs/skill-builder/` | OpenClaw agent that turns a Python package URL into a progressive-disclosure `SKILL.md` |
| **Skill-scout** | `agents/ai-skill-scout/`, `docs/skill-scout/` | OpenClaw agent that searches GitHub for existing OpenClaw skills and installs them safely |
| **Skill evaluation framework** | `agents/skill-tester/`, `docs/eval/` | Two-stage A/B framework measuring whether a skill makes an MLE agent measurably better (Stage 1 local, Stage 2 MLEvolve A/B on Nautilus NRP) |

Repo was forked from [aibuildai/AI-Build-AI](https://github.com/aibuildai/AI-Build-AI) in 2026-04, then diverged completely — fork connection severed 2026-05.

## Repo layout

```
agents/
├── ai-skill-builder/            builder agent + bundled skills
├── ai-skill-scout/              scout agent + bundled skills
└── skill-tester/                eval harness agent
docs/
├── eval/                        skill-evaluation framework (current focus)
│   ├── overview.md              2-stage pipeline summary
│   ├── stage1.md                local CI-style skill eval (locked, in production)
│   └── stage2.md                MLEvolve A/B framework (v0.2, pre-pilot)
├── skill-scout/                 OpenClaw skill-discovery agent design
│   ├── hld.md / hld.pdf
│   └── plan.md / plan.pdf
└── skill-builder/               OpenClaw skill-generator agent design
    ├── hld.md
    └── plan.md
```

When `scripts/` and `infra/` get created (tasks #73–#75), they'll hold the Stage 2 runtime: container image, k8s manifests, orchestrator.

## Skill evaluation framework — `docs/eval/`

Current research focus. Two-stage pipeline; index at `docs/eval/overview.md`.

| Stage | Question | Doc | Status |
|---|---|---|---|
| 1 | Does the skill fire on the right prompts and surface the right facts? | `docs/eval/stage1.md` | locked, in production |
| 2 | Does MLEvolve build measurably better pipelines *with* the skill, and *where* does the help land? | `docs/eval/stage2.md` | v0.2 architecture locked, pilot pending |

**Stage 2 in one sentence:** paired with-skill / without-skill A/B across PEFT-relevant tasks (LLM-Merging 5h, debug-trl-grpo 1h, jigsaw-toxic 12h deferred) on MLEvolve, decomposing trajectories into a 6 × 16 pipeline-stage taxonomy and attributing improvements via PyCG-Extended call-graphs + AST extractors + state predicates (LLM judge default OFF).

**Runtime infrastructure:** UCSD Nautilus NRP k8s cluster; one container image per (MLEvolve, deps); one Pod per (task × cell × seed) trajectory. Phase 0 → Phase 4 in `docs/eval/ops.md` (task #76).

## OpenClaw agents — `agents/`

Three agents live under `agents/`. Each follows the standard openclaw layout (`AGENTS.md` + `IDENTITY.md` + `SOUL.md` + `TOOLS.md` + `HEARTBEAT.md` + `USER.md` + `skills/` + optional `data/`).

| Agent | Doc | Role |
|---|---|---|
| `ai-skill-builder` | `docs/skill-builder/{hld,plan}.md` | Builds a `SKILL.md` from a Python package URL |
| `ai-skill-scout` | `docs/skill-scout/{hld,plan}.md` | Searches GitHub for existing OpenClaw skills |
| `skill-tester` | referenced from `docs/eval/stage1.md` | Test-harness agent (Stage 1 actually pins the `main` agent as baseline, not this one) |

These are tracked-in-repo canonical copies. The live agents run under `~/.openclaw/agents/` on the user's machine; this repo holds reproducible snapshots.

## Conventions

- **Filenames:** kebab-case (`build-skill-from-docs/`, `find-ai-skill/`). Markdown in subdirs uses short lowercase names (`hld.md`, `plan.md`, `stage1.md`).
- **Docs structure:** each multi-doc area gets a subdir under `docs/`. New agents/projects get their own subdir.
- **Skill-evaluation work always lives in `docs/eval/`** — don't scatter eval changes across stage1/stage2/agent docs.
- **Absolute paths in code/docs:** the repo lives at `/Users/kuntalkokate/pengtao-lab/ai-builds-ai-project/AI-Skill-builder/` after the 2026-05 rename. Old `/AI-Build-AI/` paths are stale.
