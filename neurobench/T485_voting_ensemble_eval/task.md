# T485_voting_ensemble_eval: Voting Ensemble Evaluation
## Task Description

Build a hard/soft voting ensemble of the top-5 models; evaluate against the best single model with bootstrap CIs.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Members selected on val only.

- Both voting rules reported.

- Save artefacts under `models/benchmark_results/T485_voting_ensemble_eval/`.


## Expected Output

Expected output artifact(s):

- `voting_metrics.csv`

- `voting_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
