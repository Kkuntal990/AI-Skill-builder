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

**MLE agent**: [MLEvolve-generic](https://github.com/e-strauss/MLEvolve-generic) (fork of AutoMLGen's MCGS search, pinned `@26bde89`), LLM `deepseek/deepseek-v4-pro` via OpenRouter. MLEvolve is the current agent; its subprocess-per-node execution avoids the fork-after-CUDA OOM that plagued the prior agent.

**Task pool** (contract-only PEFT fine-tuning tasks; same backbone `Qwen/Qwen2.5-3B-Instruct`, same unmodified `peft-tuning` skill as the treatment):

| Task | Domain | Metric | Status |
|---|---|---|---|
| `samsum` | dialogue summarization (SFT) | ROUGE-L F1 | ✅ seed 0 paired (spike-012); seed 1 running |
| `gsm8k` | math reasoning (SFT) | exact-match accuracy | instruction + held-out grader ready; see `infra/tasks/gsm8k/README.md` |
| `boolq` | yes/no QA (SFT) | accuracy (on `validation` — test labels hidden) | planned (instruction.md drafted; held-out grader entry + refs pending) |

**Metrics**:

| Layer | What it measures | How |
|---|---|---|
| **L1 outcome** | Did the skill solve the task better? | **Independent held-out grader** (`mleval.grader` recomputes the metric from the agent's preserved `submission.csv` against held-out references) + paired Lift. The agent's self-reported `Final Validation Score` is the tree-search signal + drift diagnostic only, never the headline. |
| **L2 per-sub-stage** | *Where* in the 6×16 pipeline did the skill help? | 3 co-location-proof metrics from `stage_metrics.py` — **clean-reach** (did the stage run right), **rework** (re-attempts), **failure-modes** (`exc_type` per stage) — over the multi-label AST classifier. py-spy per-stage timing + state-predicate artifact checks deferred. |
| **L3 trajectory cost** | Did the skill make the agent slower / more expensive? | Wall-clock, tokens, cost (`pricing.py`), error count |

**Pipeline stages** (6 top-level, 16 sub-stages — see [stage2.md](./stage2.md) for full taxonomy):

1. Data understanding (loading, EDA)
2. Data engineering (cleaning, splits, feature-eng)
3. Model design (architecture, loss, **adapter config**)
4. Training execution (optimizer, training loop, **preference-opt**)
5. Tuning & ablation (HPO, error-analysis)
6. Evaluation & delivery (eval, **inference/merge**, submission)

**Headline number**: mean Lift over (tasks × seeds), 95% CI from paired-seed variance.

## Token budget

**Honest range, not point estimate.** No published MLEvolve token-cost data exists. Anchored to literature peers (AIRA-dojo / AutoMLGen) and scaled to each task's per-benchmark time budget.

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

- **Stage 1**: locked, in production. Default test agent `main` for clean lift measurement; `skill-tester` for end-to-end MCP outcome plumbing. Four MCP signals captured: native tool calls, bash sidecar log, narration regex, outcome narration (Phase 1.5+ — recovers false negatives from OpenClaw's null-`toolSummary` bug on multi-step responses). Reply text preserved per trial for offline re-grading. See [stage1.md](./stage1.md).
- **Stage 2 (v0.4, MLEvolve)**: harness validated end-to-end. `samsum` seed-0 paired A/B complete (spike-012) — with-skill reached ROUGE-L 0.4331 and kept the PEFT core valid; without-skill never scored and corrupted its code into parse-error nodes. Per-sub-stage metrics (`stage_metrics.py`) + L1+L2 report (`scripts/l1_l2_compare.py`) shipped. Seed 1 running; `gsm8k` + `boolq` instruction.md authoring next. Running results tracked in [peft-skill-eval-report.md](./peft-skill-eval-report.md).

## Schedule

| When | What runs |
|---|---|
| Every `build-skill-from-docs` invocation | Stage 1 (local, automated) |
| Before promoting a skill from `workspace/skills/` to a shared registry | Stage 2 pilot first, then full sweep |
| Quarterly | Re-run Stage 1 against current frontier model; refresh saturated tasks |
