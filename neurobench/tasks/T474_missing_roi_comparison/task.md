# T474_missing_roi_comparison: Missing-ROI Robustness Comparison
## Task Description

Evaluate all models with randomly dropped ROIs (5/10/20%); compare degradation slopes.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Drop pattern shared across models per seed.

- Report graceful-degradation ranking.

- Save artefacts under `models/benchmark_results/T474_missing_roi_comparison/`.


## Expected Output

Expected output artifact(s):

- `missing_roi_metrics.csv`

- `missing_roi_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
