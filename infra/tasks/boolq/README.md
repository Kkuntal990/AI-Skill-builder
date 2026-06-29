# boolq — Yes/no reading comprehension (SFT + accuracy)

Held-out BoolQ task for MLEvolve A/B eval. Agent reads staged jsonl under
`./input/`; gold answers live in private `refs/` (post-run grader only).
BoolQ has no labelled `test` split, so the `validation` split (3,270) is the
withheld test.

## Generate and stage data

Run once on a pod with `datasets` + network (helper pod):

```bash
python3 infra/tasks/boolq/scripts/make_grading_data.py --out-root /results/data/boolq
```

This writes the agent-facing `data/{train.jsonl,test.jsonl,sample_submission.csv}`
and the private `refs/test_refs.csv` directly onto the PVC. Sync the
instruction (PVC-only; data/refs come from the generator above):

```bash
./agents/ai-skill-builder/skills/refresh-mleval-pvc/scripts/stage_task.sh boolq
```

## Recommended sweep caps

Per-run execution time is **not** in `instruction.md` — it is shown each step in
MLEvolve implementation guidelines (from `config.exec.timeout` /
`MLEVAL_EXEC_TIMEOUT_SEC`) plus the generic budget rule in
`mlevolve_sidecar/eval_harness.py`. Set caps via orchestrator:

```bash
make ab-plan TASK=boolq SEEDS="0 1" TIME_LIMIT=57600 EXEC_TIMEOUT=7200 \
  SKILL_LIBRARY=/results/skills
make ab-apply TASK=boolq SEEDS="0 1" TIME_LIMIT=57600 EXEC_TIMEOUT=7200 \
  SKILL_LIBRARY=/results/skills
```

Eval is cheap (one short generation per example), so unlike SAMSum the
bottleneck is training, not decoding. Start from the gsm8k caps and tune from
smoke: if nodes hit `exec.timeout`, raise `EXEC_TIMEOUT` (keep `TIME_LIMIT` ≥
`EXEC_TIMEOUT` + headroom for image pull + grader).

## Grader

`mleval.grader` scores `submission.csv` (`id,prediction`) against
`refs/test_refs.csv` (`id,reference_answer`) with the **accuracy** scorer
(`src/mleval/grader/accuracy.py`), which normalises `yes`/`no`/`true`/`false`.
Wired in `mleval.grader.__main__._TASKS["boolq"]` and `validate.py`.

**Deploy note:** the grader + harness live in the agent **image** (`mleval`
package, `mlevolve_sidecar/`). Rebuild + push after changing
`src/mleval/grader/*`; instruction/data changes are PVC-only.
