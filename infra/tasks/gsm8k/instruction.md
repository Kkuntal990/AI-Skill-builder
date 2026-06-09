<!--
Provenance: GSM8K (Grade School Math 8K), OpenAI.
Dataset: https://huggingface.co/datasets/openai/gsm8k (config `main`).
Paper: "Training Verifiers to Solve Math Word Problems"
(Cobbe et al., 2021). License: MIT. Not gated.

This task is harness-staged for cheap multi-step-reasoning smoke
testing on a single GPU. ~10 MB dataset, single A6000, deterministic
exact-match metric.

The recipe below is intentionally OPEN — we specify only the dataset,
model, prompt template, and output contract. Method choice, library
choice, training schedule, and inference strategy are all left to the
agent (this is what we're evaluating).
-->

## Task

Fine-tune the pre-trained instruction-tuned LLM specified below on the
GSM8K grade-school math dataset. After training, evaluate the
fine-tuned model on the test split and print the exact-match accuracy
of the final numeric answer.

## Data

- **Dataset slug**: `openai/gsm8k` on HuggingFace Hub.
- **Config**: `main` (NOT `socratic`).
- **Loader**: `datasets.load_dataset("openai/gsm8k", "main")` returns a
  `DatasetDict` with pre-built splits.
- **Splits** (already provided — DO NOT do your own train/test split):
  - `train` — 7,473 question/answer pairs
  - `test` — 1,319 pairs (use for the final exact-match accuracy)
- **Fields**: each example has `question` (the word problem, a string)
  and `answer` (the full chain-of-thought solution, a string). The
  answer ends with a final line `#### <number>` — the gold numeric
  answer is the token after `#### `. Strip any thousands-separator
  commas before comparing.
- **No `id` field**: GSM8K examples have NO identifier. For the submission,
  the `id` of an example is its **0-based row index in the `test` split**
  (i.e. `enumerate` order from `load_dataset(...)["test"]`). This order is
  fixed/deterministic — see Output contract. Do NOT shuffle the test split.

## Model

- **Backbone**: `Qwen/Qwen2.5-3B-Instruct` (3.1 B params, no gating).
- The HF cache is at `/results/.hf-cache/hf` on the mounted PVC, so
  the model weights persist across trajectories — only the first
  trajectory pays the download cost.

## Prompt template

For both training (SFT) and inference, format each example as:

    Solve the following grade-school math problem. Show your reasoning,
    then give the final answer on a new line in the form `#### <number>`.

    Problem:
    {question}

    Solution:
    {answer}

At inference time, generate after the `Solution:\n` cue and decode
until end-of-sequence (or a sensible max-new-token cap that you pick —
budget enough tokens for the multi-step reasoning AND the `#### `
line). Extract the model's predicted answer as the number following
the LAST `#### ` it emits.

## Evaluation

- **Metric**: exact-match accuracy of the predicted final number
  against the gold number, over ALL 1,319 examples in the `test`
  split. A prediction counts as correct iff its extracted integer
  equals the gold integer (compare as numbers, not strings — normalize
  commas / leading `$` / trailing `.0`).
- **No external library needed** — parse `#### ` from both the gold
  answer and the model generation yourself. Be robust to a model that
  forgets the `####` cue (treat a missing/unparseable prediction as
  wrong, do not crash).
- Generating 1,319 multi-step solutions is the dominant cost — pick a
  max-new-tokens cap and (optional) batched generation so eval fits the
  wall clock. Evaluate on the FULL test split for the reported metric.

## Output contract

Produce BOTH of the following:

1. **A submission file of per-example predictions** with EXACTLY these two
   columns and a header row:

       id,prediction

   One row for **every** one of the 1,319 `test` examples, where `id` is the
   example's **0-based row index in the `test` split** (the `enumerate` order
   of `load_dataset("openai/gsm8k", "main")["test"]`, as a string: `"0"`,
   `"1"`, …, `"1318"`) and `prediction` is your model's final integer answer.

   ⚠️ The `id` column MUST be these row indices. Do NOT hash the question,
   shuffle the test split, or use any other key — the grader keys on the
   test-split index, so mismatched ids match none of the held-out gold answers
   and the submission scores **zero** even if the answers are right. The id set
   must equal `{"0", …, "1318"}` exactly. Example rows (format only — generate
   your own predictions):

       id,prediction
       0,18
       1,3

   Save with `os.makedirs("submission", exist_ok=True)` then
   `df.to_csv("submission/submission.csv", index=False)`.

   This file is graded independently (exact-match accuracy) against held-out
   gold answers; it — not your printed number — is the trajectory's official
   score.

   ✅ **Validate the format before you finish.** A checker is provided that
   confirms your `submission.csv` is well-formed (correct `id,prediction`
   columns over the exact expected id-set: `"0"`…`"1318"`). Run it at the end of
   your script and make sure it prints `VALID`:

       import subprocess
       print(subprocess.run(
           ["python", "-m", "mleval.grader.validate", "submission/submission.csv"],
           capture_output=True, text=True).stdout)

   It checks **format only and does NOT report your score** — but a submission
   it marks `INVALID` (wrong columns, hashed/shuffled/missing ids, duplicates)
   scores **zero** when graded, however correct the answers are. Fix any
   reported issue before finishing.

2. The very last line of stdout, exactly:

       Final Validation Score: <float>

   where `<float>` is your own exact-match accuracy estimate in
   `[0.0, 1.0]`. This is your self-check and the search signal.

Save the runnable script as `runfile.py` in the working directory.
