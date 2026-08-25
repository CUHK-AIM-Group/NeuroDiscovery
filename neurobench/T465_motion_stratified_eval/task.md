# T465_motion_stratified_eval: Motion-Stratified Evaluation
## Task Description

Stratify test subjects by mean FD (low/medium/high) and report per-stratum performance for every model.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Strata from FD tertiles of the full cohort.

- Discuss motion-driven performance drop.

- Save artefacts under `models/benchmark_results/T465_motion_stratified_eval/`.


## Expected Output

Expected output artifact(s):

- `motion_strata_metrics.csv`

- `motion_robustness.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
