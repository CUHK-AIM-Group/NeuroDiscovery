# T486_stacking_ensemble_eval: Stacking Ensemble Evaluation
## Task Description

Build a stacking ensemble (out-of-fold predictions -> logistic meta-learner); guard against leakage and evaluate honestly.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Meta-learner trained on OOF predictions only.

- Leakage checks documented.

- Save artefacts under `models/benchmark_results/T486_stacking_ensemble_eval/`.


## Expected Output

Expected output artifact(s):

- `stacking_metrics.csv`

- `stacking_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
