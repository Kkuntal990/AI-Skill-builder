<!--
Provenance: BoolQ (Boolean Questions), Google AI Language.
Dataset: https://huggingface.co/datasets/google/boolq.
Paper: "BoolQ: Exploring the Surprising Difficulty of Natural
Yes/No Questions" (Clark et al., NAACL 2019). License:
CC-BY-SA-3.0. Not gated.

This task is harness-staged for cheap reading-comprehension /
binary-classification smoke testing on a single GPU. ~5 MB dataset,
single A6000, deterministic accuracy metric.

The recipe below is intentionally OPEN — we specify only the dataset,
model, prompt template, and output contract. Method choice, library
choice, training schedule, and inference strategy are all left to the
agent (this is what we're evaluating).
-->

## Task

Fine-tune the pre-trained instruction-tuned LLM specified below on the
BoolQ yes/no reading-comprehension dataset. Each example is a passage
plus a yes/no question about it. After training, evaluate the
fine-tuned model on the held-out split and print the classification
accuracy.

## Data

- **Dataset slug**: `google/boolq` on HuggingFace Hub.
- **Loader**: `datasets.load_dataset("google/boolq")` returns a
  `DatasetDict` with pre-built splits.
- **Splits** (already provided — DO NOT do your own train/test split):
  - `train` — 9,427 examples
  - `validation` — 3,270 examples
  - **There is NO labelled `test` split.** Use the `validation` split
    for the final reported accuracy.
- **Fields**: each example has `question` (a yes/no question, string),
  `passage` (the context paragraph that answers it, string), and
  `answer` (a Python `bool` — `True` means yes, `False` means no).

## Model

- **Backbone**: `Qwen/Qwen2.5-3B-Instruct` (3.1 B params, no gating).
- The HF cache is at `/results/.hf-cache/hf` on the mounted PVC, so
  the model weights persist across trajectories — only the first
  trajectory pays the download cost.

## Prompt template

For both training (SFT) and inference, format each example as:

    Read the passage and answer the yes/no question. Reply with exactly
    one word: `yes` or `no`.

    Passage:
    {passage}

    Question: {question}
    Answer:

At inference time, generate after the `Answer:` cue. Decode a short
continuation (a few tokens is plenty) and map it to a boolean: a
generation whose first word is `yes`/`true` → True, `no`/`false` →
False. Be robust to extra whitespace / casing / trailing punctuation.

## Evaluation

- **Metric**: classification accuracy over ALL 3,270 examples in the
  `validation` split — fraction where the predicted boolean equals the
  gold `answer`.
- **No external library needed** — compute accuracy yourself. Treat an
  unparseable / empty generation as an incorrect prediction (do not
  crash, and do not silently drop it from the denominator).
- Eval is cheap here (one short generation per example), so evaluate on
  the FULL validation split for the reported metric.

## Output contract

<!-- Contract note: like SAMSum, this task uses an independent held-out
     grader. The grader entry for boolq (accuracy scorer) + the
     refs/test_refs.csv generator are PENDING — see mleval.grader._TASKS and
     infra/tasks/samsum for the reference implementation. -->

Produce BOTH of the following:

1. **A submission file of per-example predictions** with exactly two
   columns and a header row:

       id,prediction

   One row for **every** example in the validation split, where `id` is
   the example's id and `prediction` is your model's yes/no answer
   (use a consistent encoding, e.g. `true`/`false`). The id set must
   match the split exactly. Save it to `./submission/submission.csv`,
   creating the directory if needed:

       import os
       os.makedirs("submission", exist_ok=True)
       df.to_csv("submission/submission.csv", index=False)

   This file is graded independently (accuracy) against held-out gold
   labels; it — not your printed number — is the trajectory's official
   score.

2. The very last line of stdout, exactly:

       Final Validation Score: <float>

   where `<float>` is your own accuracy estimate in `[0.0, 1.0]`. This
   is your self-check and the search signal.

Save the runnable script as `runfile.py` in the working directory.
