# Task template

Copy this directory to `infra/tasks/<your-task>/` and customize.

## What lives here

- `instruction.md` — the AIDE `desc_file`. Plain markdown read verbatim into AIDE's task description.
- `predicates.py` — per-task state predicates. Functions take `$MLEVAL_OUTPUT_DIR`, return bool.
- `data/` — *not in git*. Stage data onto the PVC separately (see `infra/tasks/README.md` for `kubectl cp` instructions).

## How the framework finds these files

When a Job runs, env vars point at the in-pod paths:

- `MLEVAL_TASK_INSTRUCTION_PATH=/results/data/<task>/instruction.md`
- `MLEVAL_TASK_DATA_DIR=/results/data/<task>/data`

The analyzer's `state_predicates` reads `MLEVAL_TASK_INSTRUCTION_PATH`,
walks up one dir, and tries to import `predicates.py` from there. So
`predicates.py` must also be present at `/results/data/<task>/predicates.py`.

## Data sourcing checklist

Before staging, confirm:

- [ ] License permits research/eval use.
- [ ] Data is pre-cleaned (no NaN, no leakage, no duplicate ids).
- [ ] `train.csv`/`test.csv`/`sample_submission.csv` follow the format the
      instruction.md describes (the agent will use sample_submission.csv as
      its format reference).
- [ ] Size is reasonable — full dataset fits in `<10 GB` to keep PVC sane.
      For larger datasets, subsample.
