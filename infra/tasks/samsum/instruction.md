<!--
Provenance: SAMSum (Samsung-built dialogue summarization benchmark).
Dataset: https://huggingface.co/datasets/knkarthick/samsum (mirror of
the original `samsum` dataset, which Samsung de-listed from HF Hub).
Paper: "SAMSum Corpus: A Human-annotated Dialogue Dataset for
Abstractive Summarization" (Gliwa et al., 2019).

This task is harness-staged for cheap dialogue-summarization smoke
testing on a single GPU. ~10 MB dataset, single A100/A6000,
deterministic metric.

The CONTRACT below (task family, dataset, backbone, metric, output) is
FIXED. The RECIPE is OPEN — method, library, training schedule, and
inference strategy are left to the agent (this is what we're evaluating).
-->

## Task specification

The task family, dataset, backbone, metric, and output contract below are
fixed; the recipe (method, library, training schedule, inference strategy)
is open — that is what we are evaluating.

1. **Task family** — abstractive dialogue summarization: dialogue text in →
   summary text out (a generative, text-to-text task).
2. **Dataset** — use ONLY `knkarthick/samsum` via HuggingFace `datasets`
   (splits are pre-built; do not call `train_test_split`). Do **not**
   substitute a placeholder or any other dataset (e.g. `iris` / sklearn toy
   sets); the submission is scored against the real SAMSum `test` references.
3. **Backbone** — `Qwen/Qwen2.5-3B-Instruct`.
4. **Metric & output** — mean **ROUGE-L F1** over the 819 `test` examples.
   You MUST (a) write a `submission.csv` with columns `id,generated_summary`
   and a row for **every** one of the 819 `test` ids (see Output contract) AND
   (b) print the last stdout line exactly as `Final Validation Score: <float in
   [0,1]>`. A submission with different columns, or that omits/duplicates ids,
   or covers only a subset, scores **zero**.

## Resource budget

Single GPU, with a bounded **per-run execution time limit** (you are shown the
limit and the time/steps remaining each step). Program runtime — training AND
the final test evaluation — counts toward that limit; a run that does not
finish within it is killed and scores nothing. Make sure a full
train-and-evaluate pass completes within the per-run limit.

## Task

Fine-tune the pre-trained instruction-tuned causal LM specified below on
the SAMSum dialogue summarization dataset. After training, evaluate the
fine-tuned model on the test split and print ROUGE-L F1.

## Data

- **Dataset slug**: `knkarthick/samsum` on HuggingFace Hub.
- **Loader**: `datasets.load_dataset("knkarthick/samsum")` returns a
  `DatasetDict` with pre-built splits.
- **Splits** (already provided — DO NOT do your own train/test split):
  - `train` — 14,731 dialogue/summary pairs
  - `validation` — 818 pairs (optional, for training monitoring)
  - `test` — 819 pairs (use for the final ROUGE-L)
- **Fields**: each example has `id` (a string identifier, e.g. `"13611370"`),
  `dialogue` (multi-turn chat string), and `summary` (short paraphrase).
  The `id` field is REQUIRED for the submission (see Output contract) — keep it
  alongside each example through preprocessing/inference; do not drop it.

## Model

- **Backbone**: `Qwen/Qwen2.5-3B-Instruct` (3.1 B params, no gating).
- The HF cache is at `/results/.hf-cache/hf` on the mounted PVC, so
  the model weights persist across trajectories — only the first
  trajectory pays the download cost.

## Prompt template

For both training (SFT) and inference, format each example as:

    Summarize the following dialogue:
    {dialogue}
    Summary:
    {summary}

At inference time, generate after the `Summary:\n` cue and decode
until end-of-sequence (or a sensible max-new-token cap that you pick).

## Evaluation

- **Metric**: mean ROUGE-L F1 over ALL 819 examples in the `test`
  split.
- **Library**: either `evaluate.load("rouge")` or `rouge_score` —
  both are pre-installed in the image. Use stemmer enabled.
- **DO NOT use the `validation` split for the final reported metric**
  — only `test`.

## Output contract

Produce BOTH of the following:

1. **A submission file of per-example predictions** with exactly two
   columns and a header row:

       id,generated_summary

   One row for **every** example in the `test` split (all 819), where
   `id` is the example's `id` field copied **verbatim** from the dataset
   (i.e. `str(example["id"])` — an 8-digit-style string like `13611370`)
   and `generated_summary` is your fine-tuned model's generated summary
   for that test dialogue.

   ⚠️ The `id` column MUST be the dataset's own `id` values. Do NOT hash
   the dialogue, renumber, or use the row index as the id — fabricated ids
   match none of the held-out references and the submission scores **zero**
   even if the summaries are perfect. Do not leave summaries empty. The id
   set must equal the `test` split's `id` set exactly. Example rows
   (format only — generate your own summaries):

       id,generated_summary
       13611370,Hannah needs Betty's number but Amanda doesn't have it.
       13611413,Eric and Rob are going to watch a stand-up tonight.

   Save it to `./submission/submission.csv`, creating the directory if needed:

       import os
       os.makedirs("submission", exist_ok=True)
       df.to_csv("submission/submission.csv", index=False)

   This file is graded independently against held-out reference summaries;
   it — not your printed number — is the trajectory's official score.

2. The very last line of stdout, exactly:

       Final Validation Score: <float>

   where `<float>` is your own ROUGE-L F1 estimate in `[0.0, 1.0]`. This
   is your self-check and the search signal; higher is better.

Save the runnable script as `runfile.py` in the working directory.
