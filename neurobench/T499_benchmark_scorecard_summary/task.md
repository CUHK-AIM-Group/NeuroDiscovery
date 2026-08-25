# T499_benchmark_scorecard_summary: Benchmark Scorecard Summary
## Task Description

Produce the one-page benchmark scorecard: per-model radar/summary across accuracy, robustness, fairness, efficiency, and calibration dimensions.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Dimension scores normalized 0-1 with documented scaling.

- Radar chart per model.

- Save artefacts under `models/benchmark_results/T499_benchmark_scorecard_summary/`.


## Expected Output

Expected output artifact(s):

- `scorecard.md`

- `scorecard_radars.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
