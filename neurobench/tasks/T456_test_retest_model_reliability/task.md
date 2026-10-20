# T456_test_retest_model_reliability: Model Test-Retest Reliability
## Task Description

Evaluate per-subject prediction reliability across HCP test-retest sessions for each model; rank models by ICC.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Paired sessions verified.

- ICC(2,1) + CI per model.

- Save artefacts under `models/benchmark_results/T456_test_retest_model_reliability/`.


## Expected Output

Expected output artifact(s):

- `retest_icc.csv`

- `retest_scatter.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
