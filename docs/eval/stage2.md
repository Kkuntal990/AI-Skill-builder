# Stage 2 — Skill-effect A/B framework (v0.3, AIDE pivot)

Pre-ship gate. Does `MLE-Agent + skill` build measurably better pipelines than `MLE-Agent` alone, and *where* does the help come from? Reusable for any `(agent, skill, task)` triple.

> **History.** v0.2 (Apr 2026) targeted MLEvolve as the fixed agent on MLE-Bench tasks. Discovery in May 2026: MLEvolve hard-couples to the MLE-Bench registry (its grading-server sidecar imports `mlebench.registry` + `validate_submission`), so it can't run our custom-skill A/B tasks without registering each as an MLE-Bench competition. v0.3 pivots to AIDE (WecoAI), which is task-agnostic by design and self-scores via LLM judge. MLEvolve is retained as a v0.4 secondary track for MLE-Bench tasks only.

## Experimental design

Paired with-skill / without-skill, n=3 seeds, k=2 tasks (pilot starts with k=1, scales up post-pilot). Skill availability is the *only* difference. Paired seeds for variance reduction.

Anchored to [ScienceAgentBench](https://arxiv.org/abs/2410.05080) (32.4→34.3 with-knowledge), [ML-Master](https://arxiv.org/abs/2506.16499) (with/without memory), [MLE-STAR](https://arxiv.org/abs/2506.15692) (per-block ablation). Academic precedent: [Du et al. 2026](https://arxiv.org/abs/2602.22442) — paper-only, no code.

## Agent: AIDE (primary)

[github.com/WecoAI/aideml](https://github.com/WecoAI/aideml) — predecessor of MLEvolve's MCTS design, single-file monolithic code per node (easier AST target than MLEvolve's multi-file workspaces), LLM-judge self-scoring (no external grader, no benchmark coupling).

Per-step structure: 2 LLM calls (code generation + judge). Up to `agent.steps` per trajectory; 20 default for pilot, 50 for full sweep.

**Gaps AIDE has and how we bridge them** (all four are runtime monkey-patches in `infra/agents/aide/aide_sidecar/`):

| Gap | Bridge |
|---|---|
| Discards token counts (`backend/__init__.py:67`) | `backend_wrapper` re-implements provider dispatch and captures the full 5-tuple to `prompts.jsonl` |
| Never persists prompts (logger.info only, no FileHandler) | Same wrapper writes `system_message` + `user_message` per call |
| Never calls `random.seed()`; default temp=0.5 | `seed.py` pins `random`/`numpy`/`torch`; entrypoint passes `agent.code.temp=0` + `agent.feedback.temp=0` |
| Interpreter deletes `working_dir` after each step | `interpreter_patch` snapshots `working_dir` to `$MLEVAL_OUTPUT_DIR/working_dirs/op_<step>/` before deletion |
| No native skill-injection hook | `skill_inject` monkey-patches `aide.utils.config.load_task_desc` to splice `$MLEVAL_SKILL_PATH` content into task_desc |
| Openrouter backend hardcodes `provider.order=[Fireworks]` and rejects function-calling | Image sets `OPENAI_BASE_URL=https://openrouter.ai/api/v1` so the openai backend (chat.completions path) is used |
| `report.model=gpt-4.1` default routes to OpenAI's responses.create API | Entrypoint passes `generate_report=false`; we keep `journal.json` + `tree_plot.html` which are richer anyway |

These bridges live alongside AIDE rather than forking it, so an `AIDE_REF` bump just needs a re-test of the patch surface.

## Task pool (v0.3)

Locked for the current pilot sequence. The harness-validation pilot (mvp-003, CPU) used `house-prices`. The next pilot (mvp-004 smoke + mvp-005 A/B) is GPU.

| Source | Task | Profile | Time budget | Skill paired with | Status |
|---|---|---|---|---|---|
| AIDE bundled example | `house-prices` (tabular Kaggle) | CPU | 30 min | `tabular-baseline` | mvp-003 dry-run complete (harness validation only — not PEFT) |
| [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | `llama-inference` (huggyllama mirror) | GPU 1×rtxa6000 | 30-60 min | `vllm-inference` | **next pilot** — primary skill-eval target |
| [MLE-Bench low split](https://github.com/openai/mle-bench) | TBD freeform PEFT task | GPU | 1-5 h | `peft-tuning` | follow-up after llama-inference proves the harness on GPU |

`jigsaw-toxic-comment-classification` and `debug-trl-grpo` are kept as scaffolds for the follow-up; LLM-Merging is parked pending license review.

**Full sweep target: 2 tasks × 2 cells × 3 seeds = 12 trajectories (Phase 4, #80). Current pilot: 1 task × 2 cells × 1 seed = 2 trajectories (mvp-005 A/B).**

## Three-layer metric stack

| Layer | Question | Tokens per A/B |
|---|---|---|
| **L1 outcome** | Did the skill solve the task better? | 0 (AIDE LLM-judge already part of agent budget) |
| **L2 per-stage attribution** | *Where* did the skill help? | 0 (L2a/L2b static) — **L2c judge default OFF**, +~3M if enabled |
| **L3 trajectory cost/effort** | Slower or more expensive? | 0 (parsed from `prompts.jsonl` + manifest) |

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

## Layer 1 — outcome metrics

Per-task native metric + paired Lift.

| Metric | Source |
|---|---|
| Native (val MAE / AUC / accuracy / etc.) | AIDE's per-node `metric.value` (LLM judge extracts from `Validation <metric>: <value>` stdout line) |
| Lift = `score(with) − score(without)` paired | [ScienceAgentBench](https://arxiv.org/abs/2410.05080) pattern; `mleval.analyzer.aggregate` computes it |

Headline: mean Lift over (tasks × seeds), 95% CI from paired-seed variance.

Per-trajectory "best metric" = max (or min) over all node metrics (depending on `maximize` flag), matching AIDE's own `journal.get_best_node()` behavior.

## Layer 2a — sub-stage activity (MVP via AST imports/calls)

`src/mleval/analyzer/stage_classifier.py` walks each node's code with `ast.parse`, extracts top-level imports and call names, matches against a priority-ordered rule table to assign one of the 16 sub-stages. Outputs `(top_level, sub_stage, label, confidence)` per record.

6c (`submission`) and 6b (`inference_merge`) are intentionally low-priority fallback labels (priority 10/12, confidence 0.7). Without this demotion every mvp-003 house-prices step landed in 6c because `to_csv` (always called for submission) outranked legitimate higher-stage signals like `train_test_split` + sklearn ensembles (issue #104, fix shipped 2026-05).

Pilot uses this MVP classifier. The full PyCG-Extended path (task #62) integrates [HeaderGen](https://link.springer.com/article/10.1007/s10664-024-10525-w) (95.6% precision / 95.3% recall on Kaggle notebooks) and is validated against the [Ramasamy 470-notebook corpus](https://link.springer.com/article/10.1007/s10664-022-10229-z) (target ≥80% per-stage, task #70). Pilot results are exempt from this gate; the full A/B sweep is gated on it.

## Layer 2b — choice extraction + state predicates (FREE)

**Choice extraction** is part of the MVP stage_classifier: it surfaces import names and call names per record. For pilot, "agent used LoraConfig" is captured but the *value* of `lora_rank` is not — that needs ~150 LOC of AST visitors (deferred to v0.4).

**State predicates** are deterministic Python over `working_dirs/op_<step>/` (snapshotted by `aide_sidecar.interpreter_patch`). Generic predicates in `src/mleval/analyzer/state_predicates.py`:
- `has_best_solution`, `at_least_one_non_buggy_node`, `best_metric_finite`, `prompts_log_present`.

Per-task predicates in `infra/tasks/<task>/predicates.py`:
- `submission_csv_present`, `submission_has_correct_columns`, etc.

Pattern from [AppWorld](https://arxiv.org/abs/2407.18901) + [TheAgentCompany](https://arxiv.org/abs/2412.14161). ~30 min/task to author.

## Layer 2c — evidence-grounded LLM judge (DEFAULT OFF)

Built for v0.4; default-off via config flag. Enable only if pilot's L2a + L2b fail to discriminate cells, or stakeholder report needs choice-quality dimension. Lifted `DevAsk.check` pattern from [Agent-as-a-Judge](https://github.com/metauto-ai/agent-as-a-judge/blob/main/agent_as_a_judge/module/ask.py) (Apache, ~50 LOC). Activation criteria (any apply):

- Pilot Layer 2a stage-activity deltas are < 0.05 between with-skill / without-skill cells
- Pilot Layer 2b state-predicate pass-rate deltas are similarly small
- Choice-distribution shifts are within noise

**Token cost when enabled**: ~3M total for full A/B. Conditional, not default.

## Layer 3 — trajectory cost/effort

Wall-clock (from manifest), total in/out tokens (sum of `prompts.jsonl`), error count (records with `output.errors`), per-node `req_time_sec`. Rolled up by `mleval.analyzer.aggregate`.

**13 derived metrics** computed by `mleval.analyzer.metrics` and exported per-trajectory in `report.json`:

| Metric | Definition |
|---|---|
| `cost_usd` | tokens × per-model price from `pricing.py` (OpenRouter slugs) |
| `llm_call_count` | total LLM calls (code + judge) |
| `llm_latency` | percentile breakdown of `req_time_sec` (p50/p90/max) |
| `step_exec_time` | wall time spent in `Interpreter.run` per step (snapshot ts deltas) |
| `step_count` | total AIDE steps |
| `redundant_loops` | heuristic count of code re-attempts at the same sub-stage |
| `self_correction_rate` | fraction of error-step → next-step-improves transitions |
| `hallucination` | imports/calls that fail to resolve (rough proxy via tracebacks) |
| `convergence` | step-index where best_metric first hits 95% of trajectory max |
| `first_valid_submission` | step-index where state-predicate `submission_csv_present` first fires |
| `skill_api_adoption` | fraction of skill-mentioned API symbols actually called in code |

Run-level additions (in the same module):
- `cost_normalized_lift` — Lift per dollar spent (counters "win by spending more")
- `stage_chi_square` — distributional shift in sub-stage activity between cells

Skill-citation rate (regex match of skill text in node.code) — still queued for v0.4.

## What we report per A/B cell

Output of `mleval.analyzer.aggregate` is `report.{json,md}` with:

1. **Outcome Lift** (L1) — paired with−without per (task, seed), mean + 95% CI
2. **Per-trajectory table** — task, cell, seed, status, best metric, tokens, errors, predicates passed
3. **Stage activity counts** (L2a) — count of records per sub_stage per cell
4. **State predicate pass rates** (L2b) — per-trajectory + cell mean

## Reproducibility — what we control and what we don't

Controlled:
- `random` / `numpy.random` (legacy + `default_rng`) / `torch` seeds from `$SEED` (`aide_sidecar.seed`)
- `PYTHONHASHSEED`
- LLM temperature pinned to 0
- AIDE git SHA captured in manifest (`/opt/aide/.aide_sha`)
- Image digest captured at apply time (manifest fills via pod-side env)

NOT controlled (documented limitations):
- Network-latency-driven retry counts inside AIDE's `backoff_create` change the number of `random.*` calls between steps, which then shifts what `random.shuffle(pkgs)` returns inside AIDE's `_prompt_environment`. Bounded *within* a trajectory; can diverge *across* paired runs. Worst case: temperature-0 LLM responses still differ between cells.
- AIDE's interpreter spawns a subprocess; its own RNG state is not seeded by our patches.
- Token undercount: backoff retries inside a backend's `query` are merged into a single sidecar record. L3 token totals systematically undercount actual spend by O(few %).

## Infrastructure

| Concern | Tool |
|---|---|
| Orchestration | `infra/orchestrator/run_ab.py` (pure stdlib, ~250 LOC) |
| k8s Job lifecycle | envsubst-rendered `infra/agents/aide/job.yaml.tmpl` per trajectory |
| Image | `ghcr.io/kkuntal990/mleval-agent:dev` (built on amusing) |
| Storage | `mleval-results` PVC, 1Ti CephFS RWX, `rook-cephfs` storage class |
| Cluster | UCSD Nautilus NRP, namespace `ecepxie`, 1× RTX A6000 per trajectory |
| Local analyzer | `src/mleval/analyzer/` (pip-installable as part of `mleval` package) |

**LOC budget** (current): ~70 sidecar + ~600 analyzer + ~250 orchestrator + ~80 Dockerfile/entrypoint + ~140 Job template. Plus task/skill scaffolds. Well under the v0.2 "~920 hand-written" estimate.

## File layout

See `CLAUDE.md` "Repo layout" section for the full tree. Stage-2-specific:

```
src/mleval/analyzer/        adapter_aide, stage_classifier, state_predicates, metrics, pricing, aggregate
infra/agents/aide/          Dockerfile, entrypoint.sh, run_aide.py, {job,job_cpu}.yaml.tmpl, aide_sidecar/
infra/tasks/<task>/         instruction.md, predicates.py, requirements.txt, optional data/
infra/skills/<skill>/       SKILL.md, optional references/*.md, requirements.txt
infra/orchestrator/         run_ab.py
deploy/k8s/                 pvc.yaml, helper-jupyter-1gpu.yaml, pip-warm.yaml, secret.template.yaml
```

## Execution methodology

See [`ops.md`](./ops.md) for the operational playbook (pre-flight, staging data, running the sweep, pulling results, aggregating).

**Quick reference**:

```bash
make ab-plan  TASK=mytask SEEDS="0 1" SKILL_PATH=/results/skills/peft/SKILL.md   # preview
make ab-apply TASK=mytask SEEDS="0 1" SKILL_PATH=/results/skills/peft/SKILL.md   # live
make ab-wait  TASK=mytask SEEDS="0 1" SKILL_PATH=/results/skills/peft/SKILL.md   # block
# pull results from PVC, then:
make aggregate-run RUN_DIR=./pulled-results/$MLEVAL_RUN_ID
```

## Extensibility

Plugin-shaped by design. Per-skill / per-task / per-agent bits live in composable overlays.

| Extension axis | Cost per addition |
|---|---|
| New skill | `infra/skills/<name>/SKILL.md` (~50 lines) + optional `references/` |
| New task | `infra/tasks/<name>/{instruction.md, predicates.py, data/}` (~100 LOC + data staging) |
| New agent | `infra/agents/<name>/{Dockerfile, entrypoint.sh, ...}` + `src/mleval/analyzer/adapter_<name>.py` (~300 LOC) |
| New benchmark adapter | thin wrapper around the agent's native metric (~40 LOC) |

Filesystem contracts (`trajectory.jsonl`, `prompts.jsonl`, `manifest.json`, `state.json`, `working_dirs/op_<step>/`) are the seams between agent / analyzer / reporter — nothing else couples.

## Open decisions before pilot kickoff — RESOLVED

1. **Pilot task** (#88) — `llama-inference` (MLAgentBench port). `house-prices` was used for harness validation only (mvp-003).
2. **Pilot skill** — `vllm-inference` for `llama-inference`; `tabular-baseline` was used in the mvp-003 harness validation. `peft-tuning` + `jigsaw-toxic` are staged for the follow-up.
3. **AIDE pin** — pinned at `AIDE_REF=40dcf28fc3a39e93c7192acec0c9e2e9bffa973d`. Dockerfile uses `git init + git fetch + git checkout FETCH_HEAD` so SHAs work (branches assumed by `git clone --branch` would fail on SHAs — fix in commit 41807da).

## Open decisions remaining

1. **hf-warm automation** — `make hf-warm MODEL=<name>` mirroring the `pip-warm` pattern is not yet authored. Without it, the first cohort of parallel GPU pods race-downloads checkpoints; acceptable for smoke, hurts paired A/B at scale.
2. **2nd task for full sweep** — `feedback-prize-effectiveness` from MLE-Bench low split is the working candidate; final pick gated on the llama-inference A/B's outcome.

## Roadmap

| Phase | Goal | Tasks |
|---|---|---|
| **Phase 0** — pre-flight | Cluster access, image, sidecar smoke | ✅ done (#72, #73, #84, #86) |
| **Phase 1** — pilot infra | All analyzers, orchestrator, templates, doc rewrite | ✅ done (#61, #62 MVP, #63, #64, #68, #74-#77 setup, #89) |
| **Phase 1.5** — harness shakedown | house-prices CPU dry-run (mvp-001 → mvp-002 → mvp-003) + post-pilot fixes (setsid trap, +1200s buffer, requirements.txt, openai timeout, 6c demotion, references concat, 13 derived metrics) | ✅ done (#95-#103, #110) |
| **Phase 2** — first GPU pilot | llama-inference smoke (mvp-004) + paired A/B with vllm-inference (mvp-005) | #77, #111 (gated on user approval) |
| **Phase 3** — partial A/B | 2 tasks × 2 cells × 1 seed = 4 trajectories | #79 |
| **Phase 4** — full A/B | 2 tasks × 2 cells × 3 seeds = 12 trajectories | #80 |
| **Phase 5** — reusability | Port to MLEvolve (v0.4) + 2nd skill | #67 |
| **Phase 6** — classifier upgrade | PyCG-Extended replacing MVP rule table | #62, #70 |
| **Phase 7** — L2c judge | Conditional on pilot inconclusive | #71 |

## Sources

**Pipeline-stage taxonomy & ML-code analysis**: [ScienceAgentBench](https://arxiv.org/abs/2410.05080), [HeaderGen](https://arxiv.org/abs/2301.04419) ([repo](https://github.com/secure-software-engineering/HeaderGen), [EMSE 2024](https://link.springer.com/article/10.1007/s10664-024-10525-w)), [PyCG](https://arxiv.org/abs/2103.00587) ([repo](https://github.com/vitsalis/PyCG), Apache-2.0), [DataSciBench](https://arxiv.org/abs/2502.13897), [Ramasamy EMSE 2023](https://link.springer.com/article/10.1007/s10664-022-10229-z).

**MLE-agent scaffolds**: [AIDE](https://github.com/WecoAI/aideml), [AIDE paper](https://arxiv.org/abs/2502.13138), [MLEvolve](https://github.com/InternScience/MLEvolve) (archived), [AutoMLGen](https://arxiv.org/abs/2510.08511), [AIRA-dojo](https://arxiv.org/abs/2507.02554), [ML-Master](https://arxiv.org/abs/2506.16499), [MLE-STAR](https://arxiv.org/abs/2506.15692), [RD-Agent](https://github.com/microsoft/RD-Agent).

**Eval methodology**: [Anthropic Demystifying Evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), [AppWorld](https://arxiv.org/abs/2407.18901), [TheAgentCompany](https://arxiv.org/abs/2412.14161), [Agent-as-a-Judge](https://arxiv.org/abs/2410.10934) ([repo](https://github.com/metauto-ai/agent-as-a-judge)), [Du et al. 2026](https://arxiv.org/abs/2602.22442), [RE-Bench](https://arxiv.org/abs/2411.15114), [AblationBench](https://arxiv.org/abs/2507.08038).

**Benchmarks**: [MLRC-Bench](https://arxiv.org/abs/2504.09702), [SkillsBench](https://arxiv.org/abs/2602.12670), [MLE-Bench](https://arxiv.org/abs/2410.07095) (used by archived MLEvolve track), [MLE-Dojo](https://arxiv.org/abs/2505.07782) (jigsaw source — task dropped).

**PEFT pipeline**: [HF TRL](https://huggingface.co/docs/trl), [HF PEFT](https://github.com/huggingface/peft), [Raschka 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025), [LoRA Land](https://arxiv.org/abs/2405.00732).

**Infrastructure**: [Nautilus NRP docs](https://nrp.ai/documentation), [AIRA-dojo LOGGING.md](https://github.com/facebookresearch/aira-dojo/blob/main/docs/LOGGING.md).
