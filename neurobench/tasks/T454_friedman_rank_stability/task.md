# T454_friedman_rank_stability: Friedman Rank Stability Across Settings
## Task Description

Test whether model rankings are stable across all evaluation settings (datasets x atlases): Friedman test + Kendall's W per grouping.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Groupings defined a priori.

- W interpreted with conventional bands.

- Save artefacts under `models/benchmark_results/T454_friedman_rank_stability/`.


## Expected Output

Expected output artifact(s):

- `rank_stability.csv`

- `rank_stability.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
