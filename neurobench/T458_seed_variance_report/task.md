# T458_seed_variance_report: Seed Variance Report
## Task Description

Aggregate 10-seed runs (from T422-style protocols) across models: report seed-driven variance per model as a robustness ranking.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Same seeds per model.

- Variance decomposition (seed vs. fold) where data allows.

- Save artefacts under `models/benchmark_results/T458_seed_variance_report/`.


## Expected Output

Expected output artifact(s):

- `seed_variance.csv`

- `seed_variance.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
