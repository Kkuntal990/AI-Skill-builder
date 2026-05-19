# Stage 2 — Skill-effect A/B framework (v0.2)

Pre-ship gate. Does `MLE-Agent + skill` build measurably better pipelines than `MLE-Agent` alone, and *where* does the help come from? Reusable for any `(agent, skill, task)` triple.

## Experimental design

Paired with-skill / without-skill, n=3 seeds, k=3 tasks. Skill availability is the *only* difference. Paired seeds for variance reduction.

Anchored to [ScienceAgentBench](https://arxiv.org/abs/2410.05080) (32.4→34.3 with-knowledge), [ML-Master](https://arxiv.org/abs/2506.16499) (with/without memory), [MLE-STAR](https://arxiv.org/abs/2506.15692) (per-block ablation). Academic precedent: [Du et al. 2026](https://arxiv.org/abs/2602.22442) — paper-only, no code.

## Agent: MLEvolve (fixed)

[github.com/InternScience/MLEvolve](https://github.com/InternScience/MLEvolve) — #1 on MLE-Bench (61.33% Any-Medal / 12 hr). `SearchNode.stage` typed and JSON-serialized natively. Token-tracking gap at `llm/__init__.py:64-81` — patched in ~10 LOC (task #68).

## Task pool — locked

| Benchmark | Task | Time budget | PEFT relevance |
|---|---|---|---|
| [MLE-Dojo](https://arxiv.org/abs/2505.07782) (task data only, not their Gym) | `jigsaw-toxic-comment-classification` | **12 h** | NLP fine-tuning is canonical PEFT use case |
| [MLRC-Bench](https://arxiv.org/abs/2504.09702) | `LLM-Merging` | **5 h** | Adapter merging is *the* PEFT workflow |
| [SkillsBench](https://arxiv.org/abs/2602.12670) | `debug-trl-grpo` | **1 h** (CPU-only) | LoRA-on-GRPO post-training |

**Per-task time budgets are benchmark-anchored, not arbitrary**:

- jigsaw-toxic: 12 h matches MLEvolve's SOTA-validated config; MLE-Bench standard is 24 h but jigsaw is in the `low` complexity split → converges fast; OpenAI scaling data shows diminishing returns past 24 h (medal 8.7% → 11.8% @ 24h → 100h).
- LLM-Merging: 5 h is MLRC-Bench's own `launch.sh` default (`MAX_HOURS=5`, `MAX_STEPS=50`). Competition rule itself caps merge+eval at 1 h on a 48 GB GPU.
- debug-trl-grpo: 1 h hard cap from SkillsBench `task.toml` (`agent.timeout_sec = 3600`). CPU-only (4 cores / 2 GB RAM / no GPU) — pure TRL source debugging, no training loop.

**3 tasks × 2 conditions × 3 seeds = 18 trajectories per skill A/B. Total agent runtime per seed × variant ≈ 18 hours** (was 36 h with old 12 h universal budget).

## Three-layer metric stack

| Layer | Question | Tokens per A/B |
|---|---|---|
| **L1 outcome** | Did the skill solve the task better? | 0 (deterministic graders) |
| **L2 per-stage attribution** | *Where* did the skill help? | 0 (L2a/L2b static) — **L2c judge default OFF**, +~3M if enabled |
| **L3 trajectory cost/effort** | Slower or more expensive? | 0 (parsed from logs) |

**Boundary**: L2 is per-stage only; L3 is trajectory aggregates only.

## Pipeline-stage taxonomy — 6 top-level × 16 sub-stages

| # | Top-level | Sub-stages | Anchors |
|---|---|---|---|
| 1 | Data understanding | 1a data-loading · 1b EDA | [TDSP](https://learn.microsoft.com/en-us/azure/architecture/data-science-process/), [CRISP-ML(Q)](https://ml-ops.org/content/crisp-ml), [DataSciBench](https://arxiv.org/abs/2502.13897) |
| 2 | Data engineering | 2a cleaning/encoding · 2b split & validation · 2c feature-engineering | CRISP-ML(Q), [AIRA-dojo](https://arxiv.org/abs/2507.02554) |
| 3 | Model design | 3a architecture · 3b loss · **3c adapter config `[LLM]`** | [MLE-STAR](https://arxiv.org/abs/2506.15692), [HF PEFT](https://github.com/huggingface/peft) |
| 4 | Training execution | 4a optimizer · 4b training loop · **4c preference-opt `[LLM]`** | AIRA-dojo, [Raschka 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025), [HF TRL](https://huggingface.co/docs/trl) |
| 5 | Tuning & ablation | 5a HPO · 5b ablation/error-analysis | [AblationBench](https://arxiv.org/abs/2507.08038), MLE-STAR |
| 6 | Evaluation & delivery | 6a held-out eval · **6b inference/merge `[LLM]`** · 6c submission | CRISP-ML(Q), HF PEFT `merge_and_unload` |

## Per-sub-stage metric vector (locked)

**Universal activity** (every sub-stage): `reach` (bool), `nodes_touched` (int), `code_share` (float), `tokens` (int).

**Sub-stage-specific**:

| Sub-stage | Choice extraction (AST) | Quality predicate (free) | L2c judge dimension |
|---|---|---|---|
| 1a data-loading | `loader_type`, `num_files_loaded` | `no_load_errors` | data_source_approp |
| 1b EDA | `inspection_types` (set) | `eda_output_present` | exploration_depth |
| 2a cleaning/encoding | `encoders_used` (set) | `no_nan_in_features` | preprocessing_approp |
| 2b split & validation | `split_strategy`, `n_folds` | `val_distinct_from_test` | validation_approp |
| 2c feature-engineering | `fe_types`, `num_features_added` | — | fe_relevance |
| 3a architecture | `model_family`, `model_name`, `num_params` | `model_loaded_clean` | architecture_approp |
| 3b loss/objective | `loss_class`, `custom_loss_defined` | `loss_computed_clean` | loss_approp |
| **3c adapter config** | `lora_rank`, `lora_alpha`, `target_modules`, `quantization_bits`, `use_dora` | `peft_wrap_clean` | adapter_approp |
| 4a optimizer/scheduler | `optimizer`, `learning_rate`, `has_scheduler`, `weight_decay` | — | optimizer_approp |
| 4b training loop | `epochs_run`, `batch_size`, `final_train_loss` | `training_completed` | training_setup_approp |
| **4c preference-opt** | `objective` ∈ {SFT/DPO/GRPO/SimPO/KTO} | `preference_training_completed` | objective_approp |
| 5a HPO | `hpo_method`, `num_distinct_configs` | `improvements_found` | hpo_strategy_approp |
| 5b ablation | `error_analysis_present` | — | ablation_quality |
| 6a held-out eval | `val_metric_name`, `val_metric_value` | `val_evaluation_present` | evaluation_rigor |
| **6b inference / merge** | `merge_done`, `runtime_load`, `inference_script_present` | `inference_runs_clean` | inference_correctness |
| 6c submission | `n_rows`, `n_cols`, `file_path` | `submission_present`, `format_valid` | submission_correctness |

## Layer 1 — outcome metrics

Per-task native metric + normalized + Lift.

| Metric | Source |
|---|---|
| Native (F1/AUC/REWARD/etc.) | Per-task `grade.py`, lifted from [mle-bench/grade_helpers.py](https://github.com/openai/mle-bench) (MIT) |
| HumanRank % = `1 − p/N` | [MLE-Dojo](https://arxiv.org/abs/2505.07782) formula |
| Lift = `score(with) − score(without)` paired | [ScienceAgentBench](https://arxiv.org/abs/2410.05080) pattern |

Headline: mean Lift over (tasks × seeds), 95% CI from paired-seed variance.

## Layer 2a — sub-stage activity via PyCG-Extended (FREE)

Install via `pip install --no-deps git+https://github.com/secure-software-engineering/HeaderGen.git`. Use only `pycg_extended` (call-graph) + `framework_models` (phase mappings) — not the notebook CLI. License: HeaderGen has no LICENSE file; experiment use only.

**Why PyCG-Extended over plain `ast.walk()`**: resolves aliased imports, method calls on typed objects (`trainer.train()` → `Trainer.train`), and transitive calls through helper functions. HeaderGen reports **95.6% precision / 95.3% recall** on Kaggle notebooks ([EMSE 2024](https://link.springer.com/article/10.1007/s10664-024-10525-w)).

Integration: ~30 LOC wrapper writes each `SearchNode.code` to a temp `.py`, runs `CallGraphGenerator`, maps resolved calls → HeaderGen phase → our 6×16 taxonomy. ~50 LOC PEFT signature extensions for the 6 sub-stages HeaderGen doesn't natively cover.

**Validation gate**: precision/recall against [Ramasamy 470-notebook corpus](https://link.springer.com/article/10.1007/s10664-022-10229-z) (EMSE 2023). Target ≥80% per stage. Task #70 gates the pilot.

## Layer 2b — choice extraction + state predicates (FREE)

**Choice extraction** (~150 LOC, 16 AST visitors): PyCG gives stage tags; we still need *values* (e.g., `lora_rank=16`). Walk AST for known constructor patterns.

**State predicates** (~80 LOC, ~5–8 per task): deterministic Python over playground dir, pattern from [AppWorld](https://arxiv.org/abs/2407.18901) + [TheAgentCompany](https://arxiv.org/abs/2412.14161). Examples: `submission_valid`, `checkpoint_saved`, `inference_runs_clean`. Authored per task (~30 min/task).

## Layer 2c — evidence-grounded LLM judge (BUILT but DEFAULT OFF)

Component is built for v0.2 but **disabled by default** via config flag `enable_l2c_judge: false`. Enable only if pilot's L2a + L2b fail to discriminate with-skill vs without-skill, or stakeholder report needs choice-quality dimension.

Lifted `DevAsk.check` pattern from [Agent-as-a-Judge](https://github.com/metauto-ai/agent-as-a-judge/blob/main/agent_as_a_judge/module/ask.py) (Apache, ~50 LOC). Per-sub-stage judge call with n=3 majority vote, evidence-grounded (paper shows 60-70% → 88-92% human alignment when evidence is grounded vs flat).

```text
Criterion: "Did the agent make appropriate {sub_stage} for this task?"

Evidence:
  Task: {instruction.md}
  Choices: {Layer 2b output}
  Code: {relevant snippet from contributing nodes}
  Plan: {agent's reasoning text}
  Outcome: {quality predicates}

Output: <SATISFIED> or <UNSATISFIED>, then 1-sentence reason.
```

Three samples at temperature 0.3 → `satisfied_ratio` ∈ {0/3, 1/3, 2/3, 3/3}.

**Token cost when enabled**: ~5K input + ~100 output × 3 samples × ~10 touched sub-stages × 18 trajectories ≈ **~3M tokens total for full A/B**. Negligible vs the agent budget but conditional — default OFF.

**Activation criteria** (enable if any apply):
- Pilot Layer 2a stage-activity deltas are < 0.05 between with-skill / without-skill cells (no clear discrimination)
- Pilot Layer 2b state-predicate pass-rate deltas are similarly small
- Choice-distribution shifts (LoRA rank histogram etc.) are within noise — need quality grading to distinguish "agent made *different* choices" from "agent made *better* choices"

**Validation gate**: hand-rate 20 random pilot trajectories, ≥75% per-sub-stage agreement with human. Escalation: failed sub-stages get full Agent-as-a-Judge module set.

Only the *appropriateness* axis from [Du et al. 2026](https://arxiv.org/abs/2602.22442); other 4 dims (consistency/completeness/efficiency/risk) revisited in v0.3 if signal is noisy.

## Layer 3 — trajectory cost/effort

Wall-clock, total prompt/completion/total tokens, LLM-call count, error count (sum of `is_buggy=True`), skill-citation rate (regex), score-vs-time curve ([RE-Bench](https://arxiv.org/abs/2411.15114) methodology).

## What we report per A/B cell

1. **Outcome Lift** (L1) — 1 number per task + overall, 95% CI
2. **Top-level stage activity delta** — 6 bars with-vs-without
3. **Sub-stage drill-down** — collapsible; PEFT-bold rows surfaced by `|Δ|`; choice histograms; judge `satisfied_ratio` deltas
4. **Trajectory cost** (L3) — wall-clock, tokens, errors

## Token budget

**Honest range, not point estimate.** No published MLEvolve token-cost data exists. Anchors from peers:

| Source | Reported |
|---|---|
| AIDE + o1-preview @ 24h on MLE-Bench ([arXiv:2410.07095](https://arxiv.org/abs/2410.07095)) | ~1.9M total tokens/task |
| AIDE + GPT-4-Turbo @ 24h on Weco-Kaggle ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)) | ~150K–300K tokens/task |
| AIRA-dojo operator completion tokens ([arXiv:2507.02554](https://arxiv.org/abs/2507.02554) Fig 10) | ~2× AIDE's completion-token usage |
| AutoMLGen MCGS step cap ([arXiv:2510.08511](https://arxiv.org/abs/2510.08511) § A.3) | 500-step total simulation budget |

**Per-task token ranges** (scaled to our per-task time budgets):

| Task | Time | Per-trajectory token range |
|---|---|---|
| jigsaw-toxic (12 h, GPU) | 12 h | **1.5M – 3M** |
| LLM-Merging (5 h, GPU; 50-step cap) | 5 h | **0.6M – 1.5M** |
| debug-trl-grpo (1 h, CPU) | 1 h | **0.2M – 0.5M** (no training; debug only) |

**Total token range per phase**:

| Phase | Trajectories | Total token range | Notes |
|---|---|---|---|
| Pilot (jigsaw × 2 cells × 2 seeds) | 4 | **6M – 12M** | **Primary deliverable: measure actual** |
| Full A/B (all 3 tasks × 2 × 3), L2c OFF | 18 | **14M – 30M** | Default config |
| Full A/B with L2c enabled | 18 | 14M – 30M + ~3M | Only if pilot inconclusive |
| Reusability (opt) | 18 | similar | — |

**Pilot's primary deliverable**: empirical tokens/trajectory for jigsaw under our specific MLEvolve config. Full-sweep budget locked only after pilot measurement.

**Token-reduction levers if pilot tracks high** (cited):

| Lever | Source | Expected cut |
|---|---|---|
| `use_global_memory: false` | (our analysis) | ~30% input |
| `agent.steps: 30` (default 50) | AutoMLGen 500-step cap | ~40% |
| Per-task time budget (already applied) | MLE-Bench / MLRC-Bench / SkillsBench configs | ~50% vs 12 h universal |
| `improve_failure_depth ≤ 5` | AIRA-dojo "10 nodes or 12h" debug cap | ~20% |
| `max_tokens` per call: 8K out / 50K in | MLE-Dojo § 5.3 hard caps | ~30%/call |
| Data subsampling during refinement | MLE-STAR Appendix F | ~30-50% on training-bound nodes |

Dollar cost computable on demand from current model pricing × measured tokens. Token budget is the primary metric since model choice is swappable.

**Pilot's primary deliverable**: empirical tokens/trajectory for our specific config (model + agent.steps + memory + parallelism). Full-sweep budget locked only after pilot measurement.

**Token-reduction levers if pilot tracks high** (cited):

| Lever | Source | Expected cut |
|---|---|---|
| `use_global_memory: false` | (our analysis) | ~30% input |
| `agent.steps: 30` (default 50) | AutoMLGen 500-step cap | ~40% |
| `agent.time_limit: 21600` (6h vs 12h) | KompeteAI 6h default | ~30-40% |
| `improve_failure_depth ≤ 5` | AIRA-dojo "10 nodes or 12h" debug cap | ~20% |
| `max_tokens` per call: 8K out / 50K in | MLE-Dojo § 5.3 hard caps | ~30%/call |
| Data subsampling during refinement | MLE-STAR Appendix F | ~30-50% on training-bound nodes |

Dollar cost computable on demand from current model pricing × measured tokens. Token budget is the primary metric since model choice is swappable.

## Infrastructure — pure Python, 2 external deps

No eval-framework dependency (verified [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), [OpenAI Evals](https://github.com/openai/evals), [Bloom](https://github.com/safety-research/bloom) — all overkill at 18 trajectories; no production-grade ML-pipeline-stage tracker exists per deep-search, closest is paper-only [Du et al. 2026](https://arxiv.org/abs/2602.22442)).

**Dependencies**: `pycg_extended` + `framework_models` from HeaderGen (pip --no-deps); vendored code lifts from `agent-as-a-judge/module/ask.py` (Apache) and `mle-bench/mlebench/grade_helpers.py` (MIT).

**LOC budget**: ~920 hand-written (orchestrator 120 + adapters 120 + agent integration 60 + trace adapter 50 + L2a 80 + L2b 230 + L2c 150 + L3 30 + reporter 80).

## File layout

```
ai-skill-eval-framework/
├── eval_runner.py                       # main A/B orchestrator
├── config.yaml                          # MLEvolve config + model + budgets
├── tasks/{jigsaw-toxic, llm-merging, debug-trl-grpo}/
│   └── {task.yaml, instruction.md, grade.py, Dockerfile, checkpoints.py}
├── skills/peft-tuning/                  # SKILL.md + scripts + references
├── adapters/{mle_bench, mlrc_bench, skillsbench}_score.py
├── agents/mlevolve/
│   └── {run.sh, inject_skill.py, token_patch.py, trace_adapter.py}
├── analyzer/
│   ├── layer1_outcome.py
│   ├── layer2a_stage_activity.py        # PyCG-Extended wrapper
│   ├── layer2b_choice_extractors.py     # 16 AST visitors
│   ├── layer2b_state_predicates.py
│   ├── layer2c_judge.py                 # DevAsk-pattern majority-vote
│   ├── layer3_trajectory_cost.py
│   ├── signatures/{base.py, peft.py}    # per-skill overlays
│   └── judges/peft.py                   # per-skill judge prompt
├── reporter/make_report.py
└── runs/<run_id>/<task>/<variant>/<seed>/
    └── {tree.json, tokens.jsonl, submission/, outcome.json, stages.json, judge.json, cost.json}
```

## Execution methodology

**MLEvolve**: `bash run_single_task.sh <EXP_ID> <DATASET_DIR>`. Config sets model, steps, time_limit, `use_global_memory`. Output → `runs/<ts>_<exp_id>/`.

**A/B orchestration**: paired seeds, identical config except skill flag. Resumable on failure.

**Skill injection**: single hook into `MLEvolve/agents/prompts/impl_guideline.py::get_impl_guideline()` (lines 28-60) — propagates to every operator. Plus file-mount at `/skills/{name}/`.

**Token logging patch**: ~10 LOC accumulator in `llm/__init__.py::query()` writes JSONL to `runs/<ts>_<exp_id>/tokens.jsonl`. Schema mirrors [AIRA-dojo `operators_metrics.usage`](https://github.com/facebookresearch/aira-dojo/blob/main/docs/LOGGING.md).

## Extensibility

Skill-agnostic by design. Per-skill bits live in composable overlays.

| Extension axis | Cost per addition |
|---|---|
| New skill (e.g., `vllm-serving`) | `skills/<name>/` + `analyzer/signatures/<name>.py` (~30 LOC) + `analyzer/judges/<name>.py` (~50 LOC) |
| New task | `tasks/<id>/` (~50 LOC + data) |
| New benchmark | `adapters/<bench>_score.py` (~40 LOC) |
| New MLE agent (e.g., AIDE) | `agents/<agent>/` (~150 LOC) |

Filesystem contracts (`tree.json`, `tokens.jsonl`, `submission/`) are the seams between agent / analyzer / reporter — nothing else couples.

## Implementation roadmap

| Phase | Goal | Token budget | Tasks |
|---|---|---|---|
| **Phase 0** — Classifier validation | ≥80% precision/recall vs Ramasamy 470-corpus | 0 (no agent runs) | #62 → #70 |
| **Phase 1** — Pilot | End-to-end + token calibration (L2c off) | 2M – 16M (measure actual) | #61, #63, #64, #68 → #65 |
| **Phase 2** — Full peft A/B | Headline numbers, ship decision (L2c off unless pilot inconclusive) | 9M – 75M, +~3M if L2c enabled | #66 (+#71 if enabled) |
| **Phase 3** — Reusability (opt) | Port to 2nd agent + 2nd skill | similar to Phase 2 | #67 |

## Open decisions

- **Model**: DeepSeek V4 Pro (recommend) vs Gemini-3-Pro (MLEvolve's SOTA result)
- **Pilot task**: `jigsaw-toxic` (1 task × 2 cells × 2 seeds = 4 trajectories)
- **L2c activation** (if needed post-pilot): hand-rate 20 trajectories first, ≥75% per-sub-stage agreement before relying on it

## Sources

**Pipeline-stage taxonomy & ML-code analysis**: [ScienceAgentBench](https://arxiv.org/abs/2410.05080), [HeaderGen](https://arxiv.org/abs/2301.04419) ([repo](https://github.com/secure-software-engineering/HeaderGen), [EMSE 2024](https://link.springer.com/article/10.1007/s10664-024-10525-w)), [PyCG](https://arxiv.org/abs/2103.00587) ([repo](https://github.com/vitsalis/PyCG), Apache-2.0), [DataSciBench](https://arxiv.org/abs/2502.13897), [Ramasamy EMSE 2023](https://link.springer.com/article/10.1007/s10664-022-10229-z).

**MLE-agent scaffolds**: [MLEvolve](https://github.com/InternScience/MLEvolve), [AutoMLGen](https://arxiv.org/abs/2510.08511), [AIDE](https://arxiv.org/abs/2502.13138), [AIRA-dojo](https://arxiv.org/abs/2507.02554), [ML-Master](https://arxiv.org/abs/2506.16499), [MLE-STAR](https://arxiv.org/abs/2506.15692).

**Eval methodology**: [Anthropic Demystifying Evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), [AppWorld](https://arxiv.org/abs/2407.18901), [TheAgentCompany](https://arxiv.org/abs/2412.14161), [Agent-as-a-Judge](https://arxiv.org/abs/2410.10934) ([repo](https://github.com/metauto-ai/agent-as-a-judge)), [Du et al. 2026](https://arxiv.org/abs/2602.22442), [RE-Bench](https://arxiv.org/abs/2411.15114), [AblationBench](https://arxiv.org/abs/2507.08038).

**Benchmarks**: [MLE-Dojo](https://arxiv.org/abs/2505.07782), [MLE-Bench](https://arxiv.org/abs/2410.07095), [MLRC-Bench](https://arxiv.org/abs/2504.09702), [SkillsBench](https://arxiv.org/abs/2602.12670).

**PEFT pipeline**: [HF TRL](https://huggingface.co/docs/trl), [HF PEFT](https://github.com/huggingface/peft), [Raschka 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025).

**Infrastructure**: [MLE-Bench repo](https://github.com/openai/mle-bench), [AIRA-dojo LOGGING.md](https://github.com/facebookresearch/aira-dojo/blob/main/docs/LOGGING.md).
