<!--
Provenance: SAMSum (Samsung-built dialogue summarization benchmark).
Dataset: https://huggingface.co/datasets/knkarthick/samsum (mirror of
the original `samsum` dataset, which Samsung de-listed from HF Hub).
Paper: "SAMSum Corpus: A Human-annotated Dialogue Dataset for
Abstractive Summarization" (Gliwa et al., 2019).

This task is harness-staged for cheap dialogue-summarization smoke
testing on a single GPU. ~10 MB dataset, single A100/A6000,
deterministic metric.

The recipe below is intentionally OPEN — we specify only the dataset,
model, prompt template, and output contract. Method choice, library
choice, training schedule, and inference strategy are all left to the
agent (this is what we're evaluating).
-->

## Task

Fine-tune the pre-trained instruction-tuned LLM specified below on the
SAMSum dialogue summarization dataset. After training, evaluate the
fine-tuned model on the test split and print ROUGE-L F1.

## Data

- **Dataset slug**: `knkarthick/samsum` on HuggingFace Hub.
- **Loader**: `datasets.load_dataset("knkarthick/samsum")` returns a
  `DatasetDict` with pre-built splits.
- **Splits** (already provided — DO NOT do your own train/test split):
  - `train` — 14,731 dialogue/summary pairs
  - `validation` — 818 pairs (optional, for training monitoring)
  - `test` — 819 pairs (use for the final ROUGE-L)
- **Fields**: each example has `dialogue` (multi-turn chat string) and
  `summary` (short paraphrase).

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

The very last line of stdout MUST be exactly:

    Final Validation Score: <float>

where `<float>` is the ROUGE-L F1 in `[0.0, 1.0]`. The harness parses
this line as the trajectory's metric value. Higher is better.

There is NO submission file. Save the runnable script as `runfile.py`
in the working directory.
