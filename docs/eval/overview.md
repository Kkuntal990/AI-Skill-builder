# Evaluation Plan

How we evaluate skills produced by `ai-skill-builder` and whether they make an MLE agent measurably better.

## Two-stage pipeline

| | Stage 1 — Local skill eval | Stage 2 — Skill-effect A/B framework |
|---|---|---|
| **Question** | Does the skill fire on the right prompts, surface the right facts, cite its references? | Does MLEvolve build measurably better pipelines *with* the skill, and *where* does the help land? |
| **Scope** | Skill in isolation, no MLE-agent loop | Full agent run, paired with-skill / without-skill, n=3 seeds × 3 tasks = 18 trajectories |
| **Cadence** | Every skill build (CI-style) | Pre-ship gate for a new skill |
| **Details** | [stage1.md](./stage1.md) | [stage2.md](./stage2.md) |

## Stage 2 quick reference

**MLE agent**: [MLEvolve](https://github.com/InternScience/MLEvolve) (fixed — #1 on MLE-Bench, 61.33% Any-Medal in 12 hr)

**Task pool** (1 per benchmark, each with benchmark-anchored time budget):

| Task | Benchmark | Time budget | Why PEFT-relevant |
|---|---|---|---|
| `jigsaw-toxic-comment-classification` | MLE-Dojo | **12 h** (MLEvolve SOTA config) | NLP fine-tuning is canonical PEFT use case |
| `LLM-Merging` | MLRC-Bench | **5 h** (MLRC-Bench `MAX_HOURS=5`) | Adapter merging is *the* PEFT workflow |
| `debug-trl-grpo` | SkillsBench | **1 h** CPU-only (SkillsBench hard cap) | TRL+GRPO post-training; LoRA-on-GRPO common |

**Metrics** (3 layers):

| Layer | What it measures | How |
|---|---|---|
| **L1 outcome** | Did the skill solve the task better? | Native task metric + HumanRank% + paired Lift |
| **L2 per-stage attribution** | *Where* in the 6-stage × 16-sub-stage pipeline did the skill help? | PyCG-Extended call-graph (0 tokens) + AST choice extractors (0 tokens) + state predicates (0 tokens) + evidence-grounded LLM judge (**default OFF**, +~3M tokens if enabled) |
| **L3 trajectory cost** | Did the skill make agent slower / more expensive? | Wall-clock, tokens, error count, skill-citation rate |

**Pipeline stages** (6 top-level, 16 sub-stages — see [stage2.md](./stage2.md) for full taxonomy):

1. Data understanding (loading, EDA)
2. Data engineering (cleaning, splits, feature-eng)
3. Model design (architecture, loss, **adapter config**)
4. Training execution (optimizer, training loop, **preference-opt**)
5. Tuning & ablation (HPO, error-analysis)
6. Evaluation & delivery (eval, **inference/merge**, submission)

**Headline number**: mean Lift over (tasks × seeds), 95% CI from paired-seed variance.

## Token budget

**Honest range, not point estimate.** No published MLEvolve token-cost data exists. Anchored to literature peers (AIDE / AIRA-dojo / AutoMLGen) and scaled to each task's per-benchmark time budget.

**Per-task per-trajectory range**:

| Task | Time | Token range |
|---|---|---|
| jigsaw-toxic (12 h GPU) | 12 h | **1.5M – 3M** |
| LLM-Merging (5 h GPU, 50-step cap) | 5 h | **0.6M – 1.5M** |
| debug-trl-grpo (1 h CPU) | 1 h | **0.2M – 0.5M** |

**Total per phase**:

| Phase | Trajectories | Total token range | Notes |
|---|---|---|---|
| Pilot (jigsaw only × 2 cells × 2 seeds) | 4 | **6M – 12M** | Primary deliverable: measure actual |
| Full A/B (3 tasks × 2 × 3), L2c OFF | 18 | **14M – 30M** | Default config |
| Full A/B with L2c enabled | 18 | 14M – 30M + ~3M | Only if pilot inconclusive |
| Reusability (opt) | 18 | similar | — |

Dollar cost computable on demand from current model pricing × measured tokens. Token budget is primary since model choice is swappable.

## Current status

- **Stage 1**: locked, in production. Test agent `main`, MCP sidecar capture active.
- **Stage 2 (v0.2)**: architecture and methodology locked. Implementation tasks #61–#71 pending. Pilot is the next milestone.

## Schedule

| When | What runs |
|---|---|
| Every `build-skill-from-docs` invocation | Stage 1 (local, automated) |
| Before promoting a skill from `workspace/skills/` to a shared registry | Stage 2 pilot first, then full sweep |
| Quarterly | Re-run Stage 1 against current frontier model; refresh saturated tasks |
