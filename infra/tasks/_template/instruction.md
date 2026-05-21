## Goal

<one-paragraph statement of what the agent is asked to do. State the
input modality, the prediction target, and the success criterion in
operational terms (e.g., "minimize MAE on the held-out 20% split").>

## Evaluation

Submissions are scored by <metric>. Lower is better. (Or: higher is better,
maximize=true.) The validation split is the held-out 20% of `train.csv`;
the agent should print

    Validation <metric>: <value>

as the last line of its submission script. AIDE's judge reads this line
to assign `metric.value`.

The final submission CSV must be written to `./working/submission.csv`
in the agent's working directory with the columns:

    id,prediction

## Data description

- `train.csv` — N rows. Columns: <list with units / dtype>.
- `test.csv`  — M rows. Same columns minus the target.
- `sample_submission.csv` — M rows. Format reference for `submission.csv`.

Files are pre-cleaned: no missing values, no duplicates, no leakage of
labels into features. The agent does not need to perform any data
download — everything is available under the read-only data directory.
