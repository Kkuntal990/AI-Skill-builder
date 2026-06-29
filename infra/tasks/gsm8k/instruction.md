<!--
Provenance: GSM8K (Grade School Math 8K), OpenAI.
Source: https://huggingface.co/datasets/openai/gsm8k (config `main`).
Paper: "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021).
License: MIT. Not gated.

Task-specific contract only. Shared harness rules (held-out grading, validate
tool, resource budget, submission-is-the-score) are injected via MLEvolve
implementation guidelines on every node (both A/B cells) — see
infra/tasks/_harness_rules.md (documentation mirror of
mlevolve_sidecar/eval_harness.py). They are NOT prepended to this file.
Per-run time caps are shown dynamically in those guidelines and set by the
orchestrator (`TIME_LIMIT`, `EXEC_TIMEOUT`); do not restate them here.

The TEST TARGETS ARE WITHHELD — the agent gets test questions only and
self-scores on a held-out slice of train; the full test answers live privately
in refs/test_refs.csv and are graded post-run. Files are produced by
scripts/make_grading_data.py and staged to the PVC at
/results/data/gsm8k/{data,refs}/. SYNC to the PVC before running.
-->

## Description

Grade-school math word problems with multi-step reasoning. Fine-tune the pinned
instruction-tuned causal LM on the provided training problems so that, given a
new problem, it generates a chain-of-thought solution ending in a final numeric
answer. Produce a prediction (the final integer) for every problem in the test
set. The contract below (model, data, metric, output) is FIXED; the recipe
(method, library, schedule, inference strategy) is OPEN — that is what we
evaluate.

## Dataset Description

All data is provided as files in your input directory — **do not download
GSM8K from the internet**; use only the provided files below (test targets
have been withheld on purpose).

Files (`./input/`):

- **`train.jsonl`** — 7,473 lines, one JSON object per line with keys
  `question` (the word problem) and `answer` (the full chain-of-thought
  solution; its last line is `#### <number>`, the gold final answer). Use this
  for fine-tuning AND for carving your own validation split.
- **`test.jsonl`** — 1,319 lines, one JSON object per line with keys `id`
  (a string, `"0"`…`"1318"`) and `question` **only**. There is **no `answer`
  field** — the targets are held out and graded against references you do not
  have. Predict on these; keep each `id` with its prediction.
- **`sample_submission.csv`** — the exact required output format and id-set
  (`id,prediction` with ids `"0"`…`"1318"`, empty predictions). Read this for
  column/header/id-set reference.

Load with, e.g.:

    import json
    train = [json.loads(l) for l in open("input/train.jsonl")]
    test  = [json.loads(l) for l in open("input/test.jsonl")]   # id + question, no answer

## Model

- **Backbone**: `Qwen/Qwen2.5-3B-Instruct` (3.1 B params, no gating). Load it
  with `AutoModelForCausalLM` and use **exactly this** model — the task requires
  free-form chain-of-thought *generation*, so do NOT substitute a smaller model,
  a base/non-instruct variant, or an encoder (BERT/DistilBERT cannot generate).
  The grader does not check the model; a wrong backbone surfaces only as a
  bad/invalid submission.
- The HF cache is at `/results/.hf-cache/hf` on the mounted PVC, so the weights
  persist across runs (only the first run pays the download).

## Prompt template

For both training (SFT) and inference, format each example as:

    Solve the following grade-school math problem. Show your reasoning,
    then give the final answer on a new line in the form `#### <number>`.

    Problem:
    {question}

    Solution:
    {answer}

At inference, generate after the `Solution:\n` cue and decode until
end-of-sequence (or a sensible `max_new_tokens` cap that fits the reasoning AND
the `#### ` line). Extract the prediction as the number following the **last**
`#### ` the model emits; treat a missing/unparseable answer as wrong (do not
crash).

## Evaluation

Three distinct roles — do not conflate them:

1. **Official score (post-run, not in stdout):** exact-match accuracy over
   **all 1,319 test examples**, computed by the held-out grader from
   `./submission/submission.csv` only (compare as numbers — normalise commas /
   leading `$` / trailing `.0`).

2. **Search signal (stdout during run):** hold out a slice of `train.jsonl`
   (e.g. the last ~500 lines) as a validation set. Score exact-match on **that
   slice only** and print the result as `Final Validation Score`. Do **not**
   compute this on the test set — you have no test labels.

3. **Submission artifact:** populate `./submission/submission.csv` with one
   prediction per test id. You may need a dedicated execution for full-test
   batched inference; that pass produces the graded file, not your search metric.
   Use sizeable `batch_size`, **left-padding** for this decoder-only LM, and a
   tight `max_new_tokens`; unbatched per-example decode will not finish in time.

## Submission Format

Write predictions to **`./submission/submission.csv`** (this exact path; only it
is graded), with EXACTLY these two columns and a header — matching
`input/sample_submission.csv`:

    id,prediction

One row for **every** one of the 1,319 test examples, where `id` is copied
**verbatim** from `test.jsonl` (`"0"`…`"1318"`) and `prediction` is your model's
final integer answer. The id-set must equal `{"0",…,"1318"}` exactly — do not
renumber, shuffle, or invent ids. Example rows (format only; ids are strings):

    id,prediction
    "0",18
    "1",3

Save with:

    import os
    os.makedirs("submission", exist_ok=True)
    df.to_csv("submission/submission.csv", index=False)

Then validate the format (it does NOT reveal your score):

    import subprocess
    print(subprocess.run(
        ["python", "-m", "mleval.grader.validate", "submission/submission.csv"],
        capture_output=True, text=True).stdout)

Save the runnable script as `runfile.py` in the working directory. Write
`./submission/submission.csv` as soon as a full test pass completes, so a later
step that runs out of time still leaves a gradable artifact.
