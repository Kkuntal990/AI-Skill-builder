# `infra/tasks/house-prices/`

A public Kaggle example task (House Prices: Advanced Regression
Techniques). Used as the **plumbing-validation pilot task** — purpose is to
exercise the harness end-to-end, not to validate PEFT-skill efficacy.

## Source

- Originally from Kaggle: <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>
- The dataset's original Kaggle terms permit redistribution for
  educational/research use.

## Files

| File | Source | Purpose |
|---|---|---|
| `instruction.md` | hand-written | task description — goal, evaluation contract, submission format, data dict |
| `predicates.py` | hand-written | task-specific state predicates (submission CSV checks) |
| `data/` | Kaggle House Prices dataset | `train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt` |

## Staging onto the PVC

```bash
# 1. Spin up alpine mount pod
kubectl -n $K8S_NAMESPACE run pvc-shell --rm -it --restart=Never \
    --image=alpine \
    --overrides='{"spec":{"containers":[{"name":"pvc-shell","image":"alpine","stdin":true,"tty":true,"volumeMounts":[{"name":"r","mountPath":"/results"}]}],"volumes":[{"name":"r","persistentVolumeClaim":{"claimName":"mleval-results"}}]}}' -- /bin/sh

# 2. From another terminal — copy task data + predicates + instruction
kubectl -n $K8S_NAMESPACE cp infra/tasks/house-prices/instruction.md  pvc-shell:/results/data/house-prices/instruction.md
kubectl -n $K8S_NAMESPACE cp infra/tasks/house-prices/predicates.py   pvc-shell:/results/data/house-prices/predicates.py
kubectl -n $K8S_NAMESPACE cp infra/tasks/house-prices/data            pvc-shell:/results/data/house-prices/data
```

## Why this task for the pilot

| Pro | Con |
|---|---|
| Data ships with the task — zero Kaggle auth | Tabular regression — does not exercise PEFT sub-stages (3c/4c/6b) |
| Small dataset — ~30 min wall-clock per trajectory at 20 agent steps | Stage 3a classifier rule is NN-biased; sklearn estimators may classify as `unknown` |
| Self-contained metric (RMSLE) — extractable from a single stdout line | Single-seed pilot can't compute Lift CI |

For the rationale, see [`docs/eval/ops.md`](../../../docs/eval/ops.md) and the
"plumbing validation" framing in [`docs/eval/stage2.md`](../../../docs/eval/stage2.md).
