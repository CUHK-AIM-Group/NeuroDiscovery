# T455_calibration_reliability: Classifier Calibration Analysis
## Task Description

Assess probability calibration of classifiers: reliability diagrams, ECE, Brier score per model; Platt/isotonic recalibration comparison.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Same binning across models.

- Recalibration fitted on val only.

- Save artefacts under `models/benchmark_results/T455_calibration_reliability/`.


## Expected Output

Expected output artifact(s):

- `calibration_metrics.csv`

- `reliability_diagrams.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
