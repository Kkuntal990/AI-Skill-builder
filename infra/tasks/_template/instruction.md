# <Task Name> — task instruction TEMPLATE

> Copy this dir to `infra/tasks/<task>/`, fill the `<...>` placeholders, delete
> this banner. It encodes the contract checks that, when missing, silently
> invalidated spike-018 runs (memory: project_task_instruction_authoring_checklist).
> After editing, SYNC to the PVC (`/results/data/<task>/instruction.md`) before
> running — trajectory pods read the PVC copy, not the repo/image.

Provenance: <dataset/source + citation>.

## What you must do

Fine-tune the pinned model on the dataset below and produce predictions on the
`test` split. The **backbone is FIXED**; the recipe (method, library, schedule,
inference strategy) is OPEN.

## Dataset

- **Slug / loader**: `<datasets.load_dataset("...")>` — splits are pre-built; do
  NOT make your own split.
- **Splits**: `train` — <N>; `validation` — <N> (optional); `test` — <N> (final metric).
- **Fields**: each example has **`id`** (string identifier, e.g. `"<example_id>"`),
  `<input_field>`, and `<target_field>`.
  ⚠️ List EVERY field the output contract references — especially `id`. An agent
  that doesn't see `id` here will fabricate one and score zero (see Output contract).

## Model

- **Backbone**: `<org/model-id>` (<params>, no gating). State it explicitly; the
  held-out grader does NOT check the model, so a drifted/smaller backbone surfaces
  only as a bad or invalid submission.
- HF cache is at `/results/.hf-cache/hf` on the mounted PVC (weights persist
  across trajectories).

## Evaluation

- **Metric**: <e.g. mean ROUGE-L F1> over ALL <N> `test` examples. Use `test` only.
- The **submission file** (not your printed number) is graded independently against
  held-out references — it is the trajectory's official score.

## Output contract

Produce BOTH:

1. **`./submission/submission.csv`** with EXACTLY these columns + header:

       id,<prediction_column>

   One row for **every** `test` example (all <N>), where `id` is the dataset's own
   `id` copied **verbatim** (`str(example["id"])`, e.g. `<example_id>`) and
   `<prediction_column>` is your model's prediction.

   ⚠️ The `id` column MUST be the dataset's `id` values. Do NOT hash, renumber, or
   use the row index — fabricated ids match none of the held-out references and the
   submission scores **zero** even if predictions are perfect. Example:

       id,<prediction_column>
       <example_id>,<illustrative prediction — generate your own>

   Save with `df.to_csv("submission/submission.csv", index=False)` (create the dir).

   ✅ **Validate the format before you finish.** A checker confirms your file is
   well-formed (correct columns over the exact expected id-set) — run it at the
   end and make sure it prints `VALID`:

       import subprocess
       print(subprocess.run(
           ["python", "-m", "mleval.grader.validate", "submission/submission.csv"],
           capture_output=True, text=True).stdout)

   It checks **format only and does NOT report your score**; an `INVALID` file
   scores **zero** when graded regardless of prediction quality. (For a new task,
   add its column contract to `mleval.grader.validate._TASK_COLUMNS`.)

2. The very last stdout line, exactly: `Final Validation Score: <float in [0,1]>`

## Resource notes

- Bounded per-run execution limit (shown in the agent's budget line). **Training AND
  final evaluation must finish within it** — program runtime counts toward it.
- For expensive autoregressive decoding (LLM/VLM), **batch generation** so the full
  test set fits in budget.
- (Harness-enforced: `DataLoader(num_workers=0)` on GPU — don't rely on workers.)
