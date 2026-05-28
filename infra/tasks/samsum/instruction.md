<!--
Provenance: SAMSum (Samsung-built dialogue summarization benchmark).
Dataset: https://huggingface.co/datasets/samsum (Apache-2.0).
Paper: "SAMSum Corpus: A Human-annotated Dialogue Dataset for
Abstractive Summarization" (Gliwa et al., 2019).

This task is harness-staged for cheap PEFT smoke testing: fine-tune
a 3B+ LLM with LoRA to summarize chat-style dialogues, then evaluate
ROUGE-L on the test split. ~10 MB dataset, ~15-25 min on a single
A100/A6000, deterministic metric for paired-seed A/B.

The dataset is hosted on HuggingFace Hub. The agent uses
``datasets.load_dataset('samsum')`` to fetch it; the HF cache lives at
``/results/.hf-cache/hf`` on PVC so subsequent trajectories don't
re-download.
-->

## Task

Fine-tune `meta-llama/Llama-3.2-3B-Instruct` on the **SAMSum** dialogue
summarization dataset using **LoRA** (PEFT). Train for a small number of
steps (one epoch is enough; cap iterations if needed to fit the budget),
then evaluate **ROUGE-L** on the test split. Print the final ROUGE-L F1
score as the last line of stdout.

## Evaluation

ROUGE-L F1 score (higher is better) on the SAMSum **test** split. The
score parser reads the last line of stdout, expecting exactly:

    Final Validation Score: <float>

where `<float>` is the ROUGE-L F1 in `[0.0, 1.0]`. A reasonable baseline
of LoRA-fine-tuned Llama-3.2-3B on SAMSum is around 0.40–0.45.

## Data description

- **Dataset**: HuggingFace Hub `samsum` (loaded via
  `datasets.load_dataset('samsum')`). Pre-split into:
  - `train` — 14,732 dialogue/summary pairs
  - `validation` — 818 pairs (use during training for early stopping if you wish)
  - `test` — 819 pairs (use for the final ROUGE-L)
- Each example has two fields: `dialogue` (multi-turn chat string) and
  `summary` (short paraphrase). No labels are hidden; this is a fully open
  benchmark with public test labels.
- **DO NOT do any train/test split yourself** — the HF splits are
  authoritative.
- **DO NOT use the validation split for the final score** — use `test`.

## Model

- **Backbone**: `meta-llama/Llama-3.2-3B-Instruct`.
- LoRA recipe (suggested but free to tweak): `r=16`, `alpha=32`,
  `target_modules=["q_proj","k_proj","v_proj","o_proj"]`, dropout 0.05.
- Use **fp16 or bf16** to fit the model in the GPU's memory budget
  (the spike runs on an RTX A6000 with 48 GB). Full fine-tune of a 3B
  model with optimizer state would not fit; LoRA brings the trainable
  parameter count down to ~30 MB and fits comfortably.

## Training recipe (suggested)

- Library: `trl.SFTTrainer` with `peft.LoraConfig`.
- Prompt template: `"Summarize the following dialogue:\n{dialogue}\nSummary:\n{summary}"`
  (or any equivalent — the loss should only be on the `summary` portion
  when possible, but a simpler whole-sequence loss also works for the
  smoke).
- Hyperparams: 1 epoch, lr `2e-4` with cosine schedule, batch size 4,
  gradient accumulation 4, max_seq_length 1024.

## ROUGE computation

After training, run inference on the test split with the LoRA adapter
loaded. For each test example, generate a summary (max_new_tokens ≈ 100),
strip the prompt, then compute ROUGE-L F1 against the gold summary using
either the `evaluate` library (`evaluate.load('rouge')`) or `rouge_score`
directly. Report the **mean ROUGE-L F1** across the test set.

## Output contract

The very last line of stdout must be:

    Final Validation Score: <rouge_l_f1>

where `<rouge_l_f1>` is a float in `[0.0, 1.0]`. The harness parses this
line to extract the trajectory's metric value. There is no submission
file; do not write any CSV.

The agent should save its final runnable script as `runfile.py` in the
working directory.
