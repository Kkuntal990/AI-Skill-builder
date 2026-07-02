# AI-Skill-builder

Research toolkit for **building, discovering, and evaluating OpenClaw skills** that make ML-engineering agents measurably better.

Three OpenClaw agents plus a two-stage evaluation framework that A/B-tests skills against a pluggable MLE agent (primary: [MLEvolve](https://github.com/InternScience/MLEvolve)).

## What's here

| Area | Description | Entry point |
|---|---|---|
| **Skill builder** | OpenClaw agent that turns a Python package URL into a progressive-disclosure `SKILL.md` with cited gotchas + runtime MCP fallback | [docs/skill-builder/hld.md](docs/skill-builder/hld.md) |
| **Skill scout** | OpenClaw agent that searches GitHub for existing OpenClaw skills and installs them safely | [docs/skill-scout/hld.md](docs/skill-scout/hld.md) |
| **Skill evaluation framework** | Two-stage A/B harness measuring whether a skill makes an MLE agent produce better ML pipelines, and *where* in the pipeline the help lands | [docs/eval/overview.md](docs/eval/overview.md) |

## Repo layout

```
agents/
├── ai-skill-builder/        OpenClaw agent + bundled skills
├── ai-skill-scout/          OpenClaw agent + bundled skills
└── skill-tester/            harness agent for Stage 1 evals
docs/
├── eval/
│   ├── overview.md          two-stage pipeline summary
│   ├── stage1.md            local CI-style skill eval (in production)
│   └── stage2.md            MLE-agent A/B framework (MLEvolve — current agent)
├── skill-scout/             hld.md / plan.md (+ pdfs)
└── skill-builder/           hld.md / plan.md
```

## Status

- **Skill builder**: Phase 1.4 — builds and validates skills, runtime MCP fallback wired
- **Skill scout**: Phase 1 — searches GitHub, scores, installs with security scan
- **Evaluation Stage 1**: locked, runs on every skill build (~5 min, ~$1)
- **Evaluation Stage 2 (MLE-agent A/B)**: MLEvolve is the current agent; pre-pilot. Infra runs on UCSD's Nautilus NRP k8s cluster

## License

Apache 2.0 — see [LICENSE](LICENSE).
