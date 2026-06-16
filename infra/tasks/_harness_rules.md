<!--
⚠️ NO LONGER PREPENDED TO description.md (changed 2026-06-16).

History: this file was briefly concatenated ahead of each task's instruction
(C1). That front-loaded MLEvolve's de_kaggle LLM task-cleaner with ~3 KB of
removable-genre boilerplate, which pushed it out of distribution and made it
hallucinate the whole task into garbage (spike-025), blinding the skill selector
and metric-direction. MLE-Bench+MLEvolve never hits this: MLEvolve delivers
harness info through its OWN per-node channel (agents/prompts/impl_guideline.py),
not through the task description.

OPERATIVE SOURCE NOW: the task-agnostic eval rules live as a code constant,
`_EVAL_HARNESS_RULES` in mlevolve_sidecar/skill_injector.py, injected into every
node's impl_guideline (both cells, after de_kaggle). MLEvolve's native
impl_guideline already covers resource budget / submission path / anti-cheat.

This file is retained ONLY as human-readable documentation of the rules; it is
NOT read at run time. Keep it in sync with _EVAL_HARNESS_RULES if you edit
either. Authoring guideline: docs/eval/task-authoring.md
-->

# Benchmark rules (documentation only — see header)

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

<!--
The skill-router (mlevolve_sidecar/skill_injector.py) strips everything up to
and including the END_HARNESS_RULES marker below before deciding which skills
are relevant, so these constant rules don't crowd the task-specific signal out
of the selector's context window. Keep the marker as the final line.
-->
---
<!-- END_HARNESS_RULES -->
