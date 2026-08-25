# T459_fold_variance_report: Fold Variance Report
## Task Description

Characterize fold-to-fold variance for every model: per-fold metric distributions, worst-fold analysis, and implications for reporting.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Boxplots per model.

- Worst-fold subjects exported.

- Save artefacts under `models/benchmark_results/T459_fold_variance_report/`.


## Expected Output

Expected output artifact(s):

- `fold_variance.csv`

- `fold_boxplots.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
