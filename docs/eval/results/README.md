# Stage-2 A/B — full 3-layer metric report

Snapshot of the with-skill / without-skill A/B for the **latest run of each task**, decomposed
across all three metric layers (L1 outcome · L2 stage attribution · L3 cost/effort).

| File | What |
|---|---|
| `ab-metrics-report.html` | Self-contained interactive report (open in any browser; all CSS/JS inlined) |
| `ab-metrics-report.pdf` | 8-page rendered export (A3 landscape, colour-preserved) |

## Runs covered

| Task | Run | Metric | Seeds (with / without) |
|---|---|---|---|
| boolq | `mvp-032` | accuracy | s0,s1 / s0,s1 (all valid) |
| gsm8k | `mvp-032` | exact_match | s0✗,s1,s2 / s0,s1✗,s2 |
| samsum | `spike-018` | rougeL_f | s0,s1 / s0,s1✗ |

- Agent: **MLEvolve** @ `deepseek-v4-pro`. Numbers read only from persisted `held_out_score.json` +
  `report.json` on the `mleval-results` PVC (namespace `ecepxie`), regenerated 2026-07-01.
- gsm8k-with-skill-s1 = recovered from a 17h OOM+restart; samsum with-s0 / without-s1 analyzer artifacts
  were backfilled from surviving journals. L2c LLM judge default-OFF (not run). `skill_api_adoption`
  null (these runs' manifests predate `cell.skill_path`).

## Headline

The skill trades **exploration for efficiency**: on tasks with a stable recipe (boolq, samsum) it reaches
equal-or-better score at a fraction of the cost (boolq: ~10× fewer output tokens, 2.7× cheaper, ~half the
steps/calls, far less per-stage rework); paired wins boolq-s0 **+0.132**, samsum-s0 **+0.180**. On gsm8k,
which needs search depth, the same lean push backfires (paired s2 **−0.334**: under-searches or destabilises).
All signals directional only (1–2 valid pairs/task).
