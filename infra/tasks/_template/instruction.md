<!--
<Task Name> — task instruction TEMPLATE (MLE-Bench-aligned).

Copy this dir to infra/tasks/<task>/, fill the <...> placeholders, delete this
banner. Schema + rationale: docs/eval/task-authoring.md.

This file is the TASK-SPECIFIC contract only. Shared harness rules (provided-data-only,
submission path, validate tool, held-out grading) are injected via MLEvolve
implementation guidelines on every node (both A/B cells) — see
infra/tasks/_harness_rules.md (mirror of mlevolve_sidecar/eval_harness.py).
Do NOT restate them here.

Held-out design (C3): the agent gets train (with labels) + test INPUTS ONLY
(targets stripped) + sample_submission.csv; the test targets live privately in
refs/ and are graded post-run. The search signal is a validation slice the
agent carves from train — never the test set. A scripts/make_grading_data.py
builds data/ (agent-facing) + refs/ (private) and stages them to the PVC at
/results/data/<task>/{data,refs}/. SYNC to the PVC before running — pods read
the PVC, not the repo/image.

Provenance: <dataset/source + citation>.
-->

## Description

<Task family in one line: input -> output.> Fine-tune the pinned model on the
provided training data and produce a prediction for every example in the test
set. The contract (model, data, metric, output) is FIXED; the recipe (method,
library, schedule, inference strategy) is OPEN — that is what we evaluate.

## Dataset Description

All data is provided as files in your input directory (`./input/`) — do not
download the dataset (the test targets are withheld on purpose).

- **`train.<ext>`** — <N> examples WITH targets. Use for fine-tuning AND for
  carving your own validation split.
- **`test.<ext>`** — <N> examples with **`id` + inputs ONLY** (no target field).
  Predict on these; keep each `id` with its prediction.
- **`sample_submission.csv`** — the exact output format + id-set.
- **Fields**: list EVERY field the output contract references — especially the
  **`id`** key (`<id_field>`, e.g. `"<example_id>"`). An agent that doesn't see
  `id` here will fabricate one and score zero.

## Model

- **Backbone**: `<org/model-id>` (<params>, no gating). State it explicitly and
  use exactly this model; the held-out grader does NOT check the model, so a
  drifted/smaller backbone surfaces only as a bad/invalid submission.
- HF cache is at `/results/.hf-cache/hf` (weights persist across runs).

## Evaluation

- **Official metric**: <e.g. exact-match / ROUGE-L F1> over ALL <N> test
  examples, computed by the held-out grader from your `submission.csv`.
- **Your search signal**: hold out a slice of `train.<ext>` as a validation set,
  score the metric on it yourself, and print it as `Final Validation Score`. Do
  NOT score on the test set (you have no test targets).
- <Cost note: if generation-heavy, tell them to batch / cap tokens.>

## Submission Format

Write predictions to **`./submission/submission.csv`** (this exact path; only it
is graded), EXACTLY these columns + header, matching `sample_submission.csv`:

    id,<prediction_column>

One row for **every** test example (all <N>); `id` copied **verbatim** from
`test.<ext>` (no hashing/renumber unless the task DEFINES index-as-id). Example
(format only):

    id,<prediction_column>
    <example_id>,<illustrative prediction — generate your own>

Save with `os.makedirs("submission", exist_ok=True)` then
`df.to_csv("submission/submission.csv", index=False)`, validate with
`python -m mleval.grader.validate submission/submission.csv`, and save the
runnable script as `runfile.py`. (For a new task, add its column contract to
`mleval.grader.validate._TASK_COLUMNS` and `grader.__main__._TASKS`.)
