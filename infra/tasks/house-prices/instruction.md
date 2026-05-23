## Goal

Predict the `SalePrice` for each house in `test.csv`. For each `Id`, output a
predicted sale price (positive float, USD).

## Background

This is the Kaggle "House Prices: Advanced Regression Techniques" playground
competition. The dataset describes residential homes in Ames, Iowa, with 79
explanatory variables covering location, size, materials, construction year,
amenities, and condition. The target `SalePrice` ranges roughly $35k - $755k
and has a long right tail (taking log helps with errors-on-expensive-houses).

## Evaluation

Submissions are scored by Root-Mean-Squared-Error (RMSE) between the
**logarithm** of the predicted value and the logarithm of the observed sales
price — i.e. RMSLE. Lower is better.

Hold out 20% of `train.csv` for validation. After training, your script should
print exactly one line:

```
Validation RMSLE: <float>
```

(Lowercase or different capitalization is fine — the judge parses loosely.)
This line is what the judge uses to assign the trajectory's metric.

## Submission file format

The final submission must be written to `./working/submission.csv` with a
header and the format:

```
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
...
```

One row per test-set `Id`. Use `sample_submission.csv` as a format reference.

## Data description

- `train.csv` — training set (~1,460 rows × 80 columns including `SalePrice`).
- `test.csv` — test set (~1,459 rows × 79 columns, no `SalePrice`).
- `sample_submission.csv` — format reference for `submission.csv`.
- `data_description.txt` — full prose description of every column. Read this
  for column semantics, valid categorical values, and "NA means none of" gotchas
  (e.g. `PoolQC = NA` means "no pool", not "missing data").

Files are pre-cleaned (downloaded from Kaggle, no further alteration). The
agent does not need to download anything.
