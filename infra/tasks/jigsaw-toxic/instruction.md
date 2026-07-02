## Goal

Predict the probability that each comment in `test.csv` exhibits each of six
types of toxic behavior. For each `id` in the test set, output six probabilities
in [0, 1] — one per toxicity type — into `./working/submission.csv`.

## Background

This is the Kaggle "Toxic Comment Classification Challenge" (Jigsaw + Google's
Conversation AI team). The dataset is ~159k Wikipedia talk-page comments,
human-labeled for six (non-mutually-exclusive) types of toxic behavior:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

A single comment may carry zero, one, or several labels. The labels are
imbalanced (e.g. `threat` and `identity_hate` are rare).

Comments are short-to-medium English text from Wikipedia talk-page edits. They
may be profane, vulgar, or offensive — that is the point of the task.

## Evaluation

Submissions are scored by **mean column-wise ROC AUC**: ROC AUC is computed
independently for each of the six toxicity columns, then averaged. Higher is
better. The public leaderboard reference is ~0.987 (fine-tuned BERT family).

Hold out 10–20% of `train.csv` for validation. After training, your script
should print exactly one line:

```
Validation mean ROC-AUC: <float>
```

(Capitalization is loose — the judge parses heuristically.) This line is what
the judge uses to assign the trajectory's metric.

## Submission file format

The final submission must be written to `./working/submission.csv` with a
header in **exactly this column order**:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
...
```

One row per test-set `id`. Each value is a probability in [0, 1].
`sample_submission.csv` is in this format (all values 0.5) and is a faithful
template — you may copy its structure verbatim.

Two grading gotchas worth knowing:

1. Some test-set rows are unscored (used to deter hand-labeling) — they appear
   in `test.csv` but the grader silently drops them. Do not try to detect or
   filter these from the train side; just predict for every test row.
2. The metric averages across the six columns (`average="macro"`), so getting
   all six right matters. Class imbalance means a label-frequency-aware loss
   or sampler usually outperforms vanilla BCE.

## Data description

- `train.csv` — training set (~159,571 rows; columns `id`, `comment_text`, and
  the six binary toxicity labels).
- `test.csv` — test set (~153,164 rows; columns `id`, `comment_text` only).
- `sample_submission.csv` — submission format reference, same shape as
  required `submission.csv`.

Files are pre-staged under `$MLEVAL_TASK_DATA_DIR`. The agent does not need
to download anything.

## Hardware / runtime notes

- One NVIDIA A6000 (48 GB) is available; no multi-GPU.
- Wall-clock budget is ~60 minutes per trajectory (the harness outer cap is shorter).
- The published BERT-base full-fine-tune approach takes ~20–30 min for one
  epoch on this hardware. Sub-1B encoder + PEFT (LoRA on q/v) fits in
  <8 GB and runs comparably quickly. Plan for at least a baseline + one
  iteration within the budget.
