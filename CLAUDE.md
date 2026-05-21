# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project scope

This repo houses three loosely-coupled bodies of work, all centered on **OpenClaw skills for ML-engineering agents**:

| Area | Where | What |
|---|---|---|
| **Skill-builder** | `agents/ai-skill-builder/`, `docs/skill-builder/` | OpenClaw agent that turns a Python package URL into a progressive-disclosure `SKILL.md` |
| **Skill-scout** | `agents/ai-skill-scout/`, `docs/skill-scout/` | OpenClaw agent that searches GitHub for existing OpenClaw skills and installs them safely |
| **Skill evaluation framework** | `agents/skill-tester/`, `docs/eval/` | Two-stage A/B framework measuring whether a skill makes an MLE agent measurably better (Stage 1 local, Stage 2 plug-in MLE-agent A/B on Nautilus NRP — primary agent: AIDE) |

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
│   └── stage2.md                MLE-agent A/B framework (v0.3 AIDE pivot in progress — task #87)
├── skill-scout/                 OpenClaw skill-discovery agent design
│   ├── hld.md / hld.pdf
│   └── plan.md / plan.pdf
└── skill-builder/               OpenClaw skill-generator agent design
    ├── hld.md
    └── plan.md
```

`infra/agents/<name>/` is the per-agent plugin layout: each agent gets its own Dockerfile, entrypoint, and sidecar patches. `deploy/k8s/` holds agent-agnostic manifests (PVC, helper pod, secret template).

## Skill evaluation framework — `docs/eval/`

Current research focus. Two-stage pipeline; index at `docs/eval/overview.md`.

| Stage | Question | Doc | Status |
|---|---|---|---|
| 1 | Does the skill fire on the right prompts and surface the right facts? | `docs/eval/stage1.md` | locked, in production |
| 2 | Does the MLE agent build measurably better pipelines *with* the skill, and *where* does the help land? | `docs/eval/stage2.md` | v0.3 AIDE pivot in progress |

**Stage 2 in one sentence:** paired with-skill / without-skill A/B across PEFT-relevant tasks on a pluggable MLE agent (primary: AIDE), decomposing trajectories into a 6 × 16 pipeline-stage taxonomy and attributing improvements via PyCG-Extended call-graphs + AST extractors + state predicates (LLM judge default OFF).

**Runtime infrastructure:** UCSD Nautilus NRP k8s cluster; one container image per agent plugin; one Pod per (task × cell × seed) trajectory.

## Container image — `infra/agents/<name>/`, `deploy/k8s/`

Stage 2 agent runtime image. **Build on amusing, run from this Mac.**

- **Image tag:** `ghcr.io/kkuntal990/mleval-agent:dev`
- **Base:** `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` — shared across agent plugins so ghcr.io can reuse cached layers.
- **Per-agent plugin layout:** `infra/agents/<name>/Dockerfile` + `entrypoint.sh` + optional `sidecar/` patches. Primary agent: AIDE. Build picks the agent's Dockerfile; everything else (PVC, secrets, helper pod) is shared.
- **Build/push host:** `amusing.ucsd.edu` (32-core Linux amd64, native BuildKit via per-user `~/.docker/cli-plugins/docker-buildx` v0.18.0). Mac builds via QEMU emulation work but take ~30 min vs ~5–10 min native. **Don't build locally on Mac unless amusing is down.**
- **All other ops happen from this Mac:** `kubectl` against `ecepxie`, `make k8s-*`, helper-pod lifecycle, secret creation, log tailing.
- **SSH to amusing:** `ssh ad-kkokate@amusing.ucsd.edu`. Key passphrase is the literal string `"amusing"`.
- **Config:** everything is `.env`-driven (gitignored; template at `.env.example`). `IMAGE_REGISTRY`, `IMAGE_NAME`, `IMAGE_TAG`, `K8S_NAMESPACE=ecepxie`, `GPU_TYPE=nvidia.com/rtxa6000`, `OPENROUTER_API_KEY`, `MLEVAL_LLM_MODEL`, `MLEVAL_RUN_ID`, `DEFAULT_SEED`, `GHCR_READ_TOKEN`. The Makefile `-include`s it and `envsubst`s it into the k8s manifests at apply time.
- **Manifests** in `deploy/k8s/` (agent-agnostic): `pvc.yaml` (1Ti CephFS RWX), `helper-jupyter-1gpu.yaml` (1× rtxa6000 interactive pod), `secret.template.yaml` (real Secret created via `make k8s-secret`). Per-agent Job manifests live under `infra/agents/<name>/`.
- **Hard rule (still in effect):** do **not** apply any Job to Nautilus until the user explicitly approves a live run.

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
