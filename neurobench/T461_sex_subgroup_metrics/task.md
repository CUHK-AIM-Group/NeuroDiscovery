# T461_sex_subgroup_metrics: Sex-Subgroup Metrics
## Task Description

Report per-sex performance for every model: metrics, CIs, and the male/female gap with significance.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Gaps tested with bootstrap CIs.

- Confounding by site/age discussed.

- Save artefacts under `models/benchmark_results/T461_sex_subgroup_metrics/`.


## Expected Output

Expected output artifact(s):

- `sex_subgroup_metrics.csv`

- `sex_gap_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
