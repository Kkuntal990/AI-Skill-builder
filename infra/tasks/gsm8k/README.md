# gsm8k — Grade-school math (SFT + exact-match)

Held-out GSM8K task for MLEvolve A/B eval. Agent reads staged jsonl under
`./input/`; gold answers live in private `refs/` (post-run grader only).

## Generate and stage data

Run once on a pod with `datasets` + network (helper pod):

```bash
python3 infra/tasks/gsm8k/scripts/make_grading_data.py --out-root /results/data/gsm8k
```

Sync repo → PVC (instruction only; data/refs from generator above):

```bash
./agents/ai-skill-builder/skills/refresh-mleval-pvc/scripts/stage_task.sh gsm8k
```

## Recommended sweep caps

Per-run execution time is **not** in `instruction.md` — it is shown each step in
MLEvolve implementation guidelines (from `config.exec.timeout` /
`MLEVAL_EXEC_TIMEOUT_SEC`) plus the generic budget rule in
`mlevolve_sidecar/eval_harness.py`. Set caps via orchestrator:

```bash
make ab-plan TASK=gsm8k SEEDS="0" TIME_LIMIT=5400 EXEC_TIMEOUT=2400 \
  SKILL_LIBRARY=/results/skills
make ab-apply TASK=gsm8k SEEDS="0" TIME_LIMIT=5400 EXEC_TIMEOUT=2400 \
  SKILL_LIBRARY=/results/skills
```

Tune from smoke: if nodes still hit `exec.timeout`, raise `EXEC_TIMEOUT` (keep
`TIME_LIMIT` ≥ `EXEC_TIMEOUT` + headroom for image pull + grader).

**Deploy note:** harness + `de_kaggle` fixes live in the agent **image**
(`mlevolve_sidecar/eval_harness.py`, `patches/de_kaggle.py`). Rebuild + push
after pulling those changes; instruction changes are PVC-only.

---

## Post-change verification checklist

Run after editing `instruction.md`, `eval_harness.py`, `de_kaggle.py`, or staging
data. Tick all before treating a gsm8k sweep as valid.

### Docs in sync

- [ ] `infra/tasks/gsm8k/instruction.md` — no harness prepend claim; no per-exec cap text
- [ ] `infra/tasks/_harness_rules.md` ↔ `mlevolve_sidecar/eval_harness.py` `EVAL_HARNESS_RULES`
- [ ] `docs/eval/task-authoring.md` §C1 points at `eval_harness.py` (not skill-only injector)
- [ ] `infra/tasks/_template/instruction.md` matches the same harness story

### PVC / runtime inputs

- [ ] `/results/data/gsm8k/instruction.md` on PVC matches repo (stage_task.sh)
- [ ] `/results/data/gsm8k/data/{train,test}.jsonl` + `sample_submission.csv` present
- [ ] `/results/data/gsm8k/refs/test_refs.csv` present (grader only; not in `./input/`)

### Image (after sidecar / de_kaggle change)

- [ ] Agent image rebuilt on amusing and pushed (`make docker-agent && make docker-push`)
- [ ] Dockerfile smoke passes (`eval_harness` + `de_kaggle` assertions)

### Smoke trajectory (one seed, `without_skill` — do **not** run samsum for this check)

- [ ] Agent code reads `input/train.jsonl` / `input/test.jsonl`
- [ ] Draft prompt follows task-description data loader (not “input may be empty”)
- [ ] `Final Validation Score` uses a **train holdout** (~500 rows), not 1319 test rows
- [ ] `./submission/submission.csv` written with string ids `"0"`…`"1318"`
- [ ] `python -m mleval.grader.validate submission/submission.csv` → `VALID`
- [ ] Post-run `held_out_score.json` present with finite exact-match
- [ ] `prompts.jsonl` impl_guideline includes **Resource budget** rule (from eval_harness)

### A/B validity

- [ ] Instruction text identical in `with_skill` and `without_skill` (only skill library differs)
- [ ] No skill names or “use PEFT skill” hints in `instruction.md`
