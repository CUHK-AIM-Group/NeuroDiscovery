# T450_bootstrap_ci_leaderboard: Bootstrap CI Leaderboard
## Task Description

Recompute the leaderboard with bootstrap 95% CIs (10k resamples over subject-level predictions) per model and setting.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Subject-level bootstrap, not fold-level.

- Leaderboard sorted with CI overlap discussion.

- Save artefacts under `models/benchmark_results/T450_bootstrap_ci_leaderboard/`.


## Expected Output

Expected output artifact(s):

- `leaderboard_ci.csv`

- `leaderboard_ci.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
