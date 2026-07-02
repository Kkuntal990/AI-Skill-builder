# Stage 2 — Skill-effect A/B framework (v0.5, MLEvolve spike)

Pre-ship gate. Does `MLE-Agent + skill` build measurably better pipelines than `MLE-Agent` alone, and *where* does the help come from? Reusable for any `(agent, skill, task)` triple.

> **History.**
> - **v0.2 (Apr 2026)** targeted MLEvolve. Original ditch: its grading-server sidecar imported `mlebench.registry` + `validate_submission`, blocking custom-skill A/B without registering each task as an MLE-Bench competition.
> - **v0.3 (May 2026)** pivoted to a single-file, LLM-judge MLE-agent. Worked end-to-end on tabular tasks; broke repeatedly on GPU tasks (`llama-inference`): 5 OOMs in a row (mvp-014→mvp-018) traced to fork-after-CUDA in LLM-generated `DataLoader(num_workers>0)` / `dataset.map(num_proc>0)` after `model.to('cuda')`. That agent's `multiprocessing.Process(fork)` model leaks unkillable CUDA-pinned workers; cleanup_session patches survive but +7.4 to +22.5 GiB/min memory creep OOMs the pod within minutes. It was designed and validated on Kaggle tabular; GPU users (MLE-Bench, AIRA-dojo) wrap it in nested isolation. Our setup didn't and can't (no DinD/Apptainer on Nautilus).
> - **v0.4 (May 2026)** re-evaluates MLEvolve. Re-check found the original blocker is bypassable: [`use_grading_server: false`](https://github.com/InternScience/MLEvolve/blob/26bde89/engine/validation/quality_check.py#L219) short-circuits the entire mlebench import path. Architecturally relevant: MLEvolve's `engine/executor.py` uses `subprocess.Popen` per node, with the explicit docstring [*"avoids CUDA/fork issues"*](https://github.com/InternScience/MLEvolve/blob/26bde89/engine/executor.py). The `mlevolve-smoke` branch runs the spike (this document's last section).
> - **v0.5 (Jun 2026)** fixed the outcome contract. The spike-012 control drifted off-task (solved IMDB sentiment classification) yet printed a valid-looking `Final Validation Score` the harness believed — because the metric was a self-reported stdout scalar, the one thing no credible agent benchmark uses. Fix: flip `no_submission_mode: True → False` so MLEvolve natively preserves the best node's per-example predictions (`best_submission/submission.csv`), and add an independent held-out grader (`mleval.grader`) that recomputes the metric against held-out references post-run. Drift now scores `valid:false`/~0 and drops out of the lift. Kaggle persona + `./input` framing (a drift driver) neutralized at build time by `patches/de_kaggle.py`. See **Layer 1** below.

## MLEvolve spike — current track

**Branch**: `mlevolve-smoke` (forked from `b5c7120`).

**Goal**: validate four binary claims to decide whether to port v0.4 to MLEvolve permanently.

| # | Claim | Pass criterion |
|---|---|---|
| C1 | MLEvolve's subprocess-per-step model avoids the fork-after-CUDA OOM | One full trajectory (5+ nodes) on llama-inference at 64 GiB without OOMKill |
| C2 | DeepSeek-via-OpenRouter wrapper plugs into MLEvolve's `llm.code`/`llm.feedback` | At least one successful LLM round-trip per stage, captured in `prompts.jsonl` |
| C3 | `use_grading_server: false` actually short-circuits the mle-bench coupling | `pip freeze \| grep mlebench` empty; `format_server.py` never imports |
| C4 | `journal.json` → `trajectory.jsonl` adapter stays small | ≤ 200 LoC; produces a record compatible with existing `state_predicates.py` |

**Scope (evolved)**: C1–C4 all passed on a real paired A/B. The current pilot task is `samsum` (dialogue summarization SFT), not `llama-inference` (that task is a script-optimization shape that doesn't fit MLEvolve's train/test+metric contract — see memory `project_mlevolve_contract`). Backbone `Qwen/Qwen2.5-3B-Instruct`; treatment is the unmodified `peft-tuning` skill. `samsum` seed-0 paired (spike-012) + seed-1 running.

**Image**: `ghcr.io/kkuntal990/mleval-agent:dev`. Base `vllm/vllm-openai:v0.9.2`. MLEvolve vendored as submodule pinned to `e-strauss/MLEvolve-generic@26bde89`. `mlebench` deliberately NOT installed (Dockerfile asserts absence at build time as a regression guard). Built on amusing via the **build-mleval-image** skill; trajectory pods read task/skill data from the PVC, refreshed via the **refresh-mleval-pvc** skill.

**Verdict: validated** — MLEvolve is the current agent. See `infra/agents/mlevolve/README.md` for the launch recipe.

### Per-sub-stage metrics (current — what we actually report at L2)

The L2 "where did the skill help" question is answered by **three co-location-proof per-sub-stage metrics** in `src/mleval/analyzer/stage_metrics.py`, derived from existing `trajectory.jsonl` telemetry (no new instrumentation):

| Metric | Definition (per sub-stage `s`) | Answers |
|---|---|---|
| **clean-reach** | clean nodes touching `s` ÷ all nodes touching `s` | did the agent get this stage right |
| **rework** | re-attempts beyond the first (`touches − 1`) | where it thrashed |
| **failure-modes** | `exc_type` distribution over buggy nodes touching `s` | which errors live where |

These sit on the **multi-label** classifier: `stage_classifier.py` now emits `all_sub_stages` (every stage a node's code touches) + `parse_status` (`parse_error` flags diff-patch-corrupted code that carries no classifiable stage). The adapter is `adapter_mlevolve.py` (MLEvolve `journal.json` → universal `trajectory.jsonl`). The paired L1+L2 report is `scripts/l1_l2_compare.py` (emits the 3 tables + a `stage_metrics` JSON block).

Why these three and not per-stage time/tokens: a node's script spans many sub-stages, so its `exec_time_sec`/tokens/`metric` cannot be attributed to one stage (co-location). Reachability/rework/failure-modes attribute cleanly because they need only the label-set + pass/fail status. True per-stage wall-clock (out-of-process `py-spy` sampling → AST line-range binning) and per-stage artifact predicates are designed but deferred — see the cross-domain research synthesis (tracing / process-mining / agentic-milestone literature) for the rationale. **Known gap**: `parse_error` nodes carry no labels, so per-stage rework goes blind when a cell thrashes via corrupted code — the report surfaces a trajectory-level `parse_error` count alongside.

**Known measurement caveats for the spike output**:
- `prompts.jsonl` tokens are recorded only for `llm.openai.query()` calls
  (the function-calling path). `llm.openai.generate()` (the streaming
  path used by `draft_agent`, `improve_agent`, `evolution_agent`, etc.)
  returns a bare string with no token-count metadata, so
  `in_tokens`/`out_tokens` will be `null` for those rows and the
  per-trajectory `llm_total_*_tokens` aggregate undercounts. C2 (prompt
  COUNT logging) is unaffected; only the cost-derivation accuracy is.
- `MLEVAL_PROMPTS_LOG` must be set before MLEvolve imports — the
  entrypoint exports it, but if you ever run `run_mlevolve.py` directly
  from a shell without the entrypoint wrapper, set it manually.

---

> **Note on the rest of this document.** The sections below cover the universal
> pieces (experimental design, taxonomy, three-layer metric stack, reproducibility
> framework, infrastructure) — agent-agnostic by design — alongside the
> MLEvolve-specific mechanics (sidecar patches, `journal.json` field semantics,
> RNG seed sources). The plugin-level details live in
> `infra/agents/mlevolve/README.md` and the `CLAUDE.md` "Sidecar architecture" section.

## Experimental design

Paired with-skill / without-skill, n=3 seeds, k=2 tasks (pilot starts with k=1, scales up post-pilot). Skill availability is the *only* difference. Paired seeds for variance reduction.

Anchored to [ScienceAgentBench](https://arxiv.org/abs/2410.05080) (32.4→34.3 with-knowledge), [ML-Master](https://arxiv.org/abs/2506.16499) (with/without memory), [MLE-STAR](https://arxiv.org/abs/2506.15692) (per-block ablation). Academic precedent: [Du et al. 2026](https://arxiv.org/abs/2602.22442) — paper-only, no code.

## Agent: MLEvolve

[github.com/InternScience/MLEvolve](https://github.com/InternScience/MLEvolve) — an MCGS-search MLE-agent (vendored @`26bde89`). Multi-file workspace per node, `subprocess.Popen` per-node execution (the property that avoids the fork-after-CUDA OOM), and a self-reported `Final Validation Score` search signal that we override with an external held-out grader.

Per-node structure: draft / improve / debug / evolution code-gen agents, each a code-gen LLM call, executed in a fresh subprocess. Up to `STEP_LIMIT` nodes per trajectory (5 for smoke, higher for sweeps), soft-capped by a wall-clock watchdog in `entrypoint.sh`.

**Gaps MLEvolve has and how we bridge them** (import-time monkey-patches in `infra/agents/mlevolve/mlevolve_sidecar/`, applied by `run_mlevolve.py`):

| Gap | Bridge |
|---|---|
| Never persists prompts; discards token counts | `prompt_logger` wraps `llm.openai.{query,generate}` and writes `(system, user, output, tokens, req_time)` to `prompts.jsonl` |
| Never seeds RNGs | `seed` pins `random`/`numpy`/`torch` + `PYTHONHASHSEED` from `$SEED`; LLM temperature pinned to 0 in `config.yaml` |
| LLM `determine_metric_direction` flips maximize/minimize nondeterministically | `metric_direction` pins the direction from `$MLEVAL_METRIC_MAXIMIZE` |
| Default `max_tokens` truncates long completions → corrupted SEARCH/REPLACE diffs | `token_budget` raises the cap; `diff_guard` AST-guards + normalizes the patcher |
| No native skill-injection hook | `skill_retriever` loads the skill library; `skill_injector` patches the 4 codegen agents (Tier-0 catalog + per-node temp-0 selector) |
| Kaggle persona + `./input` framing + an LLM `clean_task_desc` rewrite drift non-Kaggle tasks | `patches/de_kaggle.py` neutralizes both at build time so `description.md` reaches the agent verbatim |
| Self-reported stdout score is gameable | `eval_harness` rules + the post-run held-out grader (`mleval.grader`) recompute the metric from the preserved `submission.csv` |

These bridges patch MLEvolve at import time rather than forking it, so a submodule bump just needs a re-test of the patch surface.

## Task pool (v0.3)

Locked for the current pilot sequence. The harness-validation pilot (mvp-003, CPU) used `house-prices`. The next pilot (mvp-004 smoke + mvp-005 A/B) is GPU.

| Source | Task | Profile | Time budget | Skill paired with | Status |
|---|---|---|---|---|---|
| Public Kaggle example | `house-prices` (tabular Kaggle) | CPU | 30 min | `tabular-baseline` | mvp-003 dry-run complete (harness validation only — not PEFT) |
| [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | `llama-inference` (huggyllama mirror) | GPU 1×rtxa6000 | 30-60 min | `vllm-inference` | **next pilot** — primary skill-eval target |
| [MLE-Bench low split](https://github.com/openai/mle-bench) | TBD freeform PEFT task | GPU | 1-5 h | `peft-tuning` | follow-up after llama-inference proves the harness on GPU |

`jigsaw-toxic-comment-classification` and `debug-trl-grpo` are kept as scaffolds for the follow-up; LLM-Merging is parked pending license review.

**Full sweep target: 2 tasks × 2 cells × 3 seeds = 12 trajectories (Phase 4, #80). Current pilot: 1 task × 2 cells × 1 seed = 2 trajectories (mvp-005 A/B).**

## Three-layer metric stack

| Layer | Question | Tokens per A/B |
|---|---|---|
| **L1 outcome** | Did the skill solve the task better? | 0 (MLEvolve's self-scoring already part of agent budget) |
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

Per-task metric + paired Lift, computed by an **independent held-out grader** — not the agent's self-report.

**Two-level design** (faithful to mle-bench):

- *Search signal (gameable, internal):* MLEvolve ranks nodes by the agent's self-reported `Final Validation Score` (LLM-parsed from stdout into `metric.value`) and picks its best node — exactly as MLEvolve's own best-node selection does. This drives the search but is **not** the reported outcome.
- *Headline (trustworthy, external):* with `no_submission_mode: False`, MLEvolve preserves the best node's per-example predictions at `best_submission/submission.csv`. After the run, `entrypoint.sh` invokes `mleval.grader`, which recomputes the metric from that file against held-out references and writes `held_out_score.json`. `mleval.analyzer.aggregate` uses that as the trajectory's `best_metric` (self-reported value kept only as a drift diagnostic).

| Metric | Source |
|---|---|
| Native held-out score (ROUGE-L / exact-match / accuracy) | `held_out_score.json` — `mleval.grader` over the agent's preserved `submission.csv` |
| Lift = `score(with) − score(without)` paired | [ScienceAgentBench](https://arxiv.org/abs/2410.05080) pattern; `mleval.analyzer.aggregate` computes it |

Headline: mean Lift over (tasks × seeds), 95% CI from paired-seed variance.

**Why external grading:** a self-reported scalar is gameable — a drifted trajectory (e.g. solving IMDB instead of SAMSum) can print a high number on the *wrong* task, which the old stdout-`Final Validation Score` contract accepted. The held-out grader keys on the real task's test ids and scores the output type, so drift lands as `valid:false` / ~0 and **drops out of the lift** instead of inflating it. This mirrors mle-bench: it grades one agent-selected `submission.csv` once against private answers and never feeds the agent a test score (the in-run server is format-only). The agent's self-selection of "best" is imperfect on purpose — the validation→test gap is an accepted benchmark property, not something we paper over with live test feedback.

## Layer 2a — sub-stage activity (MVP via AST imports/calls)

`src/mleval/analyzer/stage_classifier.py` walks each node's code with `ast.parse`, extracts top-level imports and call names, matches against a priority-ordered rule table to assign one of the 16 sub-stages. Outputs `(top_level, sub_stage, label, confidence)` per record.

6c (`submission`) and 6b (`inference_merge`) are intentionally low-priority fallback labels (priority 10/12, confidence 0.7). Without this demotion every mvp-003 house-prices step landed in 6c because `to_csv` (always called for submission) outranked legitimate higher-stage signals like `train_test_split` + sklearn ensembles (issue #104, fix shipped 2026-05).

Pilot uses this MVP classifier. The full PyCG-Extended path (task #62) integrates [HeaderGen](https://link.springer.com/article/10.1007/s10664-024-10525-w) (95.6% precision / 95.3% recall on Kaggle notebooks) and is validated against the [Ramasamy 470-notebook corpus](https://link.springer.com/article/10.1007/s10664-022-10229-z) (target ≥80% per-stage, task #70). Pilot results are exempt from this gate; the full A/B sweep is gated on it.

## Layer 2b — choice extraction + state predicates (FREE)

**Choice extraction** is part of the MVP stage_classifier: it surfaces import names and call names per record. For pilot, "agent used LoraConfig" is captured but the *value* of `lora_rank` is not — that needs ~150 LOC of AST visitors (deferred to v0.4).

**State predicates** are deterministic Python over the trajectory's preserved artifacts (`mlevolve_runs/<ts>/` workspaces + the best node's `submission.csv`). Generic predicates in `src/mleval/analyzer/state_predicates.py`:
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
| `step_count` | total search steps (nodes) |
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
- `random` / `numpy.random` (legacy + `default_rng`) / `torch` seeds from `$SEED` (`mlevolve_sidecar.seed`)
- `PYTHONHASHSEED`
- LLM temperature pinned to 0
- MLEvolve submodule SHA captured in manifest (`agent.version = vendored-26bde89`)
- Image digest captured at apply time (manifest fills via pod-side env)
- **Frozen package versions**: the curated `infra/agents/mlevolve/requirements.txt` resolved once at image build inside `vllm/vllm-openai:v0.9.2`. No runtime pip install at trajectory startup (the prior per-task pip install architecture was abandoned 2026-05-25 after vllm 0.6.6 in task reqs forced torch 2.7→2.5 downgrade — see commit 72cb6bd). A `pip freeze` of the running container is snapshotted to `pip_freeze.txt` in the output dir for audit.

NOT controlled (documented limitations):
- LLM sampling + network-latency-driven retry counts vary the RNG-call sequence between steps. Bounded *within* a trajectory; can diverge *across* paired runs. Worst case: temperature-0 LLM responses still differ between cells.
- MLEvolve executes each node in a subprocess; the subprocess inherits `PYTHONHASHSEED`, but any RNG re-seeding inside the generated code is not controlled by our patches.
- Token undercount: backoff retries inside a backend's `query` are merged into a single sidecar record. L3 token totals systematically undercount actual spend by O(few %).

**Tool surface — MLEvolve has no MCP client; benchmark policies vary.** Our harness inherits MLEvolve's tool surface: the LLM emits Python, the executor runs it in a subprocess, captured stdout/stderr feeds the next call. MLEvolve's function-calling use is limited to structured-output extraction (code review, result parsing, leakage checks) — no general tool dispatch, no MCP client, no `web_search`. We previously claimed MLE-Bench bans MCP/web access; on careful re-read, this overstates the source:

- **MLE-Bench paper** (arxiv 2410.07095 §2.3.1) enumerates exactly three bans: hand-labeling the submission, viewing other people's Kaggle solutions (plagiarism), and calling another external LLM API. Internet access is a *reportable variable*, not a banned one (§2.3). MCP, web browsing, and doc-lookup are not named.
- **MLE-Bench agent-facing instructions** ([`environment/instructions.txt`](https://github.com/openai/mle-bench/blob/main/environment/instructions.txt)) restate only two of those three rules (no hand-labeling, no plagiarism). The LLM-API ban is *not* shown to the agent.
- **MLE-Bench container** ([`environment/Dockerfile`](https://github.com/openai/mle-bench/blob/main/environment/Dockerfile), [`container_configs/default.json`](https://github.com/openai/mle-bench/blob/main/environment/config/container_configs/default.json)) has no `network_mode: none`, no firewall, no DNS scrub. Outbound HTTP works by construction; `curl`/`wget`/`git` are pre-installed.
- **MLE-Bench post-hoc auditor** ([`extras/rule_violation_detector/prompts.py`](https://github.com/openai/mle-bench/blob/main/extras/rule_violation_detector/prompts.py)) runs gpt-4o-mini offline over completed logs with three rubric questions. `UNAUTHORIZED_ACCESS_QUESTION` mentions "such as using a web browser" as one example, but this is advisory — judged after the run, not blocked at runtime.
- **MLAgentBench** ResearchAgent has a closed hand-coded action set (List/Read/Write/Edit/Execute/Reflect) with no web/MCP primitive — restriction by design rather than rule.
- **DSBench** ships offline bundles with code execution only.

So the "no MCP" reality is MLEvolve-specific (and broadly true of MLAgentBench and DSBench) rather than universally rule-enforced. The MLE-Bench-only fact is the LLM-API ban; everything else (web, MCP, doc-lookup) is permitted by container policy. For this harness specifically, `vllm-inference`'s `context7__*` MCP fallback hooks are unreachable because MLEvolve has no MCP client, not because MLE-Bench forbids them. The bundled `references/*.md` content carries the load the MCP fallbacks would otherwise fetch — which is why progressive-disclosure skills front-load deep material into references rather than relying solely on runtime lookup. *Future work*: enabling MCP for MLEvolve via an HTTP-shim sidecar around `mcp.context7.com` is tracked separately and would let the skill's MCP fallback path actually fire.

## Infrastructure

| Concern | Tool |
|---|---|
| Orchestration | `infra/orchestrator/run_ab.py` (pure stdlib, ~250 LOC) |
| k8s Job lifecycle | envsubst-rendered `infra/agents/mlevolve/job.yaml.tmpl` per trajectory |
| Image | `ghcr.io/kkuntal990/mleval-agent:dev` (single-image, MLE-Bench-style; base `vllm/vllm-openai:v0.9.2`; deps from `requirements.txt`; built on amusing) |
| Storage | `mleval-results` PVC, 1Ti CephFS RWX, `rook-cephfs` storage class |
| Cluster | UCSD Nautilus NRP, namespace `ecepxie`, 1× RTX A6000 per trajectory |
| Local analyzer | `src/mleval/analyzer/` (pip-installable as part of `mleval` package) |

**LOC budget** (current): ~70 sidecar + ~600 analyzer + ~250 orchestrator + ~80 Dockerfile/entrypoint + ~140 Job template. Plus task/skill scaffolds. Well under the v0.2 "~920 hand-written" estimate.

## File layout

See `CLAUDE.md` "Repo layout" section for the full tree. Stage-2-specific:

```
src/mleval/analyzer/        adapter_mlevolve, stage_classifier, state_predicates, metrics, pricing, aggregate
infra/agents/mlevolve/      Dockerfile, entrypoint.sh, run_mlevolve.py, job.yaml.tmpl, config.yaml, mlevolve_sidecar/, patches/, upstream/
infra/tasks/<task>/         instruction.md, predicates.py, requirements.txt, optional data/
infra/skills/<skill>/       SKILL.md, optional references/*.md, optional scripts/*, requirements.txt
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
| New skill | `infra/skills/<name>/SKILL.md` (~50-150 lines) + optional `references/*.md` (surfaced into the prompt by `skill_injector`) + optional `scripts/*` |
| New task | `infra/tasks/<name>/{instruction.md, predicates.py, data/}` (~100 LOC + data staging) |
| New agent | `infra/agents/<name>/{Dockerfile, entrypoint.sh, ...}` + `src/mleval/analyzer/adapter_<name>.py` (~300 LOC) |
| New benchmark adapter | thin wrapper around the agent's native metric (~40 LOC) |

Filesystem contracts (`trajectory.jsonl`, `prompts.jsonl`, `manifest.json`, `state.json`, `working_dirs/op_<step>/`) are the seams between agent / analyzer / reporter — nothing else couples.

## Open decisions before pilot kickoff — RESOLVED

1. **Pilot task** (#88) — `llama-inference` (MLAgentBench port). `house-prices` was used for harness validation only (mvp-003).
2. **Pilot skill** — `vllm-inference` for `llama-inference`; `tabular-baseline` was used in the mvp-003 harness validation. `peft-tuning` + `jigsaw-toxic` are staged for the follow-up.
3. **MLEvolve pin** — vendored as a git submodule at `infra/agents/mlevolve/upstream/` pinned to SHA `26bde89`; `make docker-mlevolve` runs `git submodule update --init` then COPYs the tree into the image. Bump the submodule to upgrade.

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

**MLE-agent scaffolds**: [MLEvolve](https://github.com/InternScience/MLEvolve), [AutoMLGen](https://arxiv.org/abs/2510.08511), [AIRA-dojo](https://arxiv.org/abs/2507.02554), [ML-Master](https://arxiv.org/abs/2506.16499), [MLE-STAR](https://arxiv.org/abs/2506.15692), [RD-Agent](https://github.com/microsoft/RD-Agent).

**Eval methodology**: [Anthropic Demystifying Evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), [AppWorld](https://arxiv.org/abs/2407.18901), [TheAgentCompany](https://arxiv.org/abs/2412.14161), [Agent-as-a-Judge](https://arxiv.org/abs/2410.10934) ([repo](https://github.com/metauto-ai/agent-as-a-judge)), [Du et al. 2026](https://arxiv.org/abs/2602.22442), [RE-Bench](https://arxiv.org/abs/2411.15114), [AblationBench](https://arxiv.org/abs/2507.08038).

**Benchmarks**: [MLRC-Bench](https://arxiv.org/abs/2504.09702), [SkillsBench](https://arxiv.org/abs/2602.12670), [MLE-Bench](https://arxiv.org/abs/2410.07095) (used by archived MLEvolve track), [MLE-Dojo](https://arxiv.org/abs/2505.07782) (jigsaw source — task dropped).

**PEFT pipeline**: [HF TRL](https://huggingface.co/docs/trl), [HF PEFT](https://github.com/huggingface/peft), [Raschka 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025), [LoRA Land](https://arxiv.org/abs/2405.00732).

**Infrastructure**: [Nautilus NRP docs](https://nrp.ai/documentation), [AIRA-dojo LOGGING.md](https://github.com/facebookresearch/aira-dojo/blob/main/docs/LOGGING.md).
