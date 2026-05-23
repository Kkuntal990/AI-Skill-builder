---
name: tabular-baseline
description: Sensible defaults for small-to-mid scale tabular regression / classification with mixed dtypes
version: 0.1.0
references:
  - title: LightGBM documentation
    url: https://lightgbm.readthedocs.io/
  - title: scikit-learn preprocessing guide
    url: https://scikit-learn.org/stable/modules/preprocessing.html
---

# Tabular baseline

## When to use this skill

On structured tabular regression or classification with:

- Mixed dtypes (numerics + categoricals).
- Row count under ~100,000 (otherwise consider distributed training).
- A clear validation metric to optimize against.

Do **not** use this skill for image, text, audio, time-series, or
fine-tuning-of-pretrained-models tasks — those need their own skills.

## Quick checklist

- [ ] **Load** with `pandas.read_csv`; inspect dtypes and missingness with `.info()` and `.isna().sum()`.
- [ ] **Hold out** 20% of the training data for validation (`sklearn.model_selection.train_test_split` with `random_state` set).
- [ ] **Encode categoricals**: one-hot for low-cardinality (< 20 unique), target-encode for high-cardinality.
- [ ] **Scale numerics**: `StandardScaler` is the safe default; tree-based models don't need it but it doesn't hurt.
- [ ] **Fit a LightGBM baseline first** with default parameters. Report the validation metric.
- [ ] Only after the LightGBM baseline is logged, consider Random Forest, XGBoost, or linear models for comparison.
- [ ] **Write the submission** with the exact columns the task spec asks for.

## Why these defaults

LightGBM is the strongest single-model baseline on most small-to-mid tabular
problems: it handles mixed dtypes natively (including categoricals if encoded
as `category` dtype), is robust to outliers, requires little hyperparameter
tuning to be competitive, and is fast (~seconds on 1k-100k rows).

Holding out 20% gives a stable validation estimate without losing too much
training data. With a small dataset, switch to 5-fold CV before relying on
the validation number to decide between models.

One-hot encoding low-cardinality categoricals plays well with all linear and
tree-based models. Target encoding is appropriate when a categorical has so
many unique values that one-hot would blow up the feature count — but only
if the target is known at training time (which it always is here).

## Common pitfalls

- **Data leakage from preprocessing**: fit `StandardScaler` / encoders on
  the **training fold only**, then transform validation and test. Do not fit
  on the combined train+test set.
- **Log-target with negative predictions**: if the task asks for RMSLE and
  the model can output negatives, clip predictions to be > 0 before
  computing log.
- **`SalePrice = NA`-as-categorical trap**: in some Kaggle datasets, `NA` in
  a categorical column means a category (e.g., "no pool") rather than
  missing data. Always read the data dictionary before imputing.
- **Forgetting to write the submission in the exact required format**:
  re-read the task spec for column names and write `./working/submission.csv`.

## Minimal example

```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

train = pd.read_csv("./input/train.csv")
test  = pd.read_csv("./input/test.csv")

# Separate target, encode categoricals as 'category' dtype (LightGBM handles them).
y = np.log1p(train["SalePrice"])
X = train.drop(columns=["SalePrice", "Id"])
for col in X.select_dtypes("object").columns:
    X[col] = X[col].astype("category")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = lgb.LGBMRegressor(random_state=0)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

val_pred = model.predict(X_val)
val_rmsle = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSLE: {val_rmsle:.5f}")

# Submission
X_test = test.drop(columns=["Id"])
for col in X_test.select_dtypes("object").columns:
    X_test[col] = X_test[col].astype("category")
test_pred = np.expm1(model.predict(X_test))
test_pred = np.clip(test_pred, a_min=0, a_max=None)
pd.DataFrame({"Id": test["Id"], "SalePrice": test_pred}).to_csv(
    "./working/submission.csv", index=False
)
```
