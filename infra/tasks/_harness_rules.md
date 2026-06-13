<!--
Shared benchmark rules, prepended to EVERY task's instruction.md by
infra/agents/mlevolve/entrypoint.sh (concatenated into prepared/public/
description.md, ahead of the task-specific contract).

This is the analogue of MLE-Bench's environment/instructions.txt: the FIXED,
task-agnostic rules. Anything task-specific (backbone, dataset, metric, exact
columns) belongs in the per-task instruction.md, NOT here.

Stage to the PVC at /results/data/_harness_rules.md (one level above each
<task>/ dir). If absent, the entrypoint falls back to the task instruction
alone — so existing tasks are unaffected until this file is staged.

Authoring guideline: docs/eval/task-authoring.md
-->

# Benchmark rules (read first)

These rules apply to every task and **override the task description below if
they ever conflict**.

1. **Provided data only.** Use exclusively the files in your input directory
   (e.g. `train.*`, `test.*`, `sample_submission.csv`). Do **not** download the
   dataset from the internet or reconstruct held-out labels from any external
   source. We have constructed our own train/test split; the test inputs you are
   given have their targets withheld — they are graded against references you do
   not have access to.

2. **Build a model from the data — do not fabricate answers.** Predictions must
   come from a model you train/fine-tune on the provided training data. Do not
   hand-write, look up, or copy predictions, and do not train or select on the
   held-out test set.

3. **One graded artifact, one fixed path.** Write your predictions to the
   submission path named in the task below and **nowhere else** — only that file
   is graded. Match the columns and id-set of `sample_submission.csv` exactly;
   it is the canonical format. A submission with wrong columns, missing/extra/
   duplicate ids, or fabricated ids scores **zero**, no matter how good the
   predictions are.

4. **Validate format before you finish.** A checker is provided:

       python -m mleval.grader.validate <your-submission>.csv

   It reports `VALID` / `INVALID` (format only — it does **not** reveal your
   score). Fix any `INVALID` before finishing.

5. **The submission file is your official score — your printed number is not.**
   The last stdout line `Final Validation Score: <float>` is only your *search
   signal* (used to guide the agent's own iteration). Estimate it on a
   validation split you hold out from the training data — **never** on the test
   set. Your real score is computed independently, after the run, from the
   submission file against held-out references.

6. **Resource budget is fixed.** Each run has a per-execution wall-clock cap
   (shown each step). Both training **and** the final test-set evaluation must
   finish within it; an unfinished run is killed and leaves only whatever
   submission it had already written. Budget your schedule accordingly, and for
   generation-heavy evaluation, batch your decoding.

---
