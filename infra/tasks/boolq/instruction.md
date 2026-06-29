<!--
Provenance: BoolQ (Boolean Questions), Google AI Language.
Source: https://huggingface.co/datasets/google/boolq.
Paper: "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No
Questions" (Clark et al., NAACL 2019). License: CC-BY-SA-3.0. Not gated.

Task-specific contract only. Shared harness rules (held-out grading, validate
tool, resource budget, submission-is-the-score) are injected via MLEvolve
implementation guidelines on every node (both A/B cells) — see
infra/tasks/_harness_rules.md (documentation mirror of
mlevolve_sidecar/eval_harness.py). They are NOT prepended to this file.
Per-run time caps are shown dynamically in those guidelines and set by the
orchestrator (`TIME_LIMIT`, `EXEC_TIMEOUT`); do not restate them here.

The TEST TARGETS ARE WITHHELD — the agent gets test questions+passages only and
self-scores on a held-out slice of train; the full test answers live privately
in refs/test_refs.csv and are graded post-run. BoolQ ships no labelled `test`
split, so its `validation` split (3,270) is used as the withheld test. Files are
produced by scripts/make_grading_data.py and staged to the PVC at
/results/data/boolq/{data,refs}/. SYNC to the PVC before running.
-->

## Description

Yes/no reading-comprehension over short passages. Fine-tune the pinned
instruction-tuned causal LM on the provided training examples so that, given a
passage and a yes/no question about it, it answers correctly. Produce a yes/no
prediction for every example in the test set. The contract below (model, data,
metric, output) is FIXED; the recipe (method, library, schedule, inference
strategy) is OPEN — that is what we evaluate.

## Dataset Description

All data is provided as files in your input directory — **do not download BoolQ
from the internet**; use only the provided files below (test targets have been
withheld on purpose).

Files (`./input/`):

- **`train.jsonl`** — 9,427 lines, one JSON object per line with keys
  `question` (a yes/no question, string), `passage` (the context paragraph that
  answers it, string), and `answer` (a JSON boolean — `true` means yes, `false`
  means no). Use this for fine-tuning AND for carving your own validation split.
- **`test.jsonl`** — 3,270 lines, one JSON object per line with keys `id`
  (a string, `"0"`…`"3269"`), `question`, and `passage` **only**. There is **no
  `answer` field** — the targets are held out and graded against references you
  do not have. Predict on these; keep each `id` with its prediction.
- **`sample_submission.csv`** — the exact required output format and id-set
  (`id,prediction` with ids `"0"`…`"3269"`, empty predictions). Read this for
  column/header/id-set reference.

Load with, e.g.:

    import json
    train = [json.loads(l) for l in open("input/train.jsonl")]
    test  = [json.loads(l) for l in open("input/test.jsonl")]   # id + question + passage, no answer

## Model

- **Backbone**: `Qwen/Qwen2.5-3B-Instruct` (3.1 B params, no gating). Load it
  with `AutoModelForCausalLM` and use **exactly this** model — answer the
  question by *generating* `yes`/`no` after the prompt cue, so do NOT substitute
  a smaller model or a base/non-instruct variant. The grader does not check the
  model; a wrong backbone surfaces only as a bad/invalid submission.
- The HF cache is at `/results/.hf-cache/hf` on the mounted PVC, so the weights
  persist across runs (only the first run pays the download).

## Prompt template

For both training (SFT) and inference, format each example as:

    Read the passage and answer the yes/no question. Reply with exactly
    one word: `yes` or `no`.

    Passage:
    {passage}

    Question: {question}
    Answer:

At inference, generate after the `Answer:` cue. Decode a short continuation (a
few tokens is plenty) and map it to a boolean: a generation whose first word is
`yes`/`true` → yes, `no`/`false` → no. Be robust to extra whitespace / casing /
trailing punctuation; treat an unparseable/empty generation as a (wrong)
prediction — do not crash and do not drop it.

## Evaluation

Three distinct roles — do not conflate them:

1. **Official score (post-run, not in stdout):** classification accuracy over
   **all 3,270 test examples**, computed by the held-out grader from
   `./submission/submission.csv` only (the predicted yes/no compared to the gold
   answer; the scorer normalises `yes`/`no`/`true`/`false`).

2. **Search signal (stdout during run):** hold out a slice of `train.jsonl`
   (e.g. the last ~1,000 lines) as a validation set. Score accuracy on **that
   slice only** and print the result as `Final Validation Score`. Do **not**
   compute this on the test set — you have no test labels.

3. **Submission artifact:** populate `./submission/submission.csv` with one
   prediction per test id. Eval is light here (one short generation per example),
   but you still have 3,270 of them — use a sizeable `batch_size` and
   **left-padding** for this decoder-only LM with a tight `max_new_tokens`;
   unbatched per-example decode wastes the budget.

## Submission Format

Write predictions to **`./submission/submission.csv`** (this exact path; only it
is graded), with EXACTLY these two columns and a header — matching
`input/sample_submission.csv`:

    id,prediction

One row for **every** one of the 3,270 test examples, where `id` is copied
**verbatim** from `test.jsonl` (`"0"`…`"3269"`) and `prediction` is your model's
yes/no answer in a consistent encoding (e.g. `yes`/`no` or `true`/`false`). The
id-set must equal `{"0",…,"3269"}` exactly — do not renumber, shuffle, or invent
ids. Example rows (format only; ids are strings):

    id,prediction
    "0",yes
    "1",no

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
