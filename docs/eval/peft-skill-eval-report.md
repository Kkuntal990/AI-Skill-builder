# PEFT-tuning skill — A/B eval report

Running report for the `peft-tuning` skill evaluated on the MLEvolve agent.
Filled in cell-by-cell as seeds/tasks complete. Final accuracy is **not** the
target — the signal is per-sub-stage uplift (clean-reach · rework · failure-modes).

- **Agent**: MLEvolve (vendored 26bde89), LLM `deepseek/deepseek-v4-pro`
- **Skill (treatment)**: `peft-tuning` (method-general LoRA/PEFT; unmodified across tasks)
- **Backbone**: `Qwen/Qwen2.5-3B-Instruct`
- **Metrics**: 3 per-sub-stage (`stage_metrics.py`) + L1/L2 (`l1_l2_compare.py`)

## Status matrix

| Task | seed | with_skill | without_skill |
|---|---|---|---|
| samsum | 0 | ✅ done (wall-capped, metric=0.4331) | ✅ done (no metric) |
| samsum | 1 | ⏳ queued | ⏳ queued |
| samsum | 2 | — | — |
| gsm8k | 0–2 | — | — |
| boolq | 0–2 | — | — |

---

## samsum · seed 0 (run_id `mlevolve-spike-012`)

Paired smoke that validated the pipeline. Image: pre-`#190` (analyzer chain was
killed by watchdog; manifests written post-hoc — fixed for seed 1+).

**Headline:** with the skill the agent locked onto the PEFT core, reworked it
until one run passed (ROUGE-L 0.4331); without it, the agent broke pre-training
and corrupted its own code into 4 parse-error nodes, never scoring.

| | with_skill | without_skill |
|---|---|---|
| Best ROUGE-L F1 | **0.4331** | none (0 successes) |
| Exit | 143 wall-capped (succeeded first) | 0 completed (step budget) |
| Total nodes (incl root) | 5 | 6 |
| parse_error nodes | 0 | **4** |
| Cost (USD) | 0.020 | 0.047 (2.4×) |
| LLM wall-time (s) | 1132 | 2613 (2.3×) |
| Self-correction rate | 0.33 | 0.00 |

**Per-sub-stage (3 metrics):**

| Stage | clean-reach W / N | rework W / N | fail-modes W / N |
|---|---|---|---|
| 1a load | 0.25 / 0.00 | 3 / 0 | RuntimeError×3 / AttributeError×1 |
| 3c adapter (PEFT) | 0.25 / 0.00 | 3 / 0 | RuntimeError×3 / AttributeError×1 |
| 4b train | 0.25 / 0.00 | 3 / 0 | RuntimeError×3 / AttributeError×1 |
| 6b infer | 0.25 / 0.00 | 3 / 0 | RuntimeError×3 / AttributeError×1 |

> without_skill's per-stage rework reads 0 because its 4 retry nodes were
> parse_error (no classifiable stage) — its thrash is real but stage-invisible.
> The 4-parse-error count is the truer "it thrashed" signal here.

Artifacts: `pulled-results/mlevolve-spike-012/` (`_l1_l2_report.{md,json}`, `_stage_metrics.md`).
