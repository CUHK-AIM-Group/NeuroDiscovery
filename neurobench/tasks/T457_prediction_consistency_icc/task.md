# T457_prediction_consistency_icc: Prediction Consistency Across Folds
## Task Description

For models with out-of-fold predictions, compute prediction consistency across fold re-assignments (multi-seed re-split) as ICC.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- 3 re-splits minimum.

- Per-model ICC + ranking.

- Save artefacts under `models/benchmark_results/T457_prediction_consistency_icc/`.


## Expected Output

Expected output artifact(s):

- `fold_consistency.csv`

- `consistency_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
