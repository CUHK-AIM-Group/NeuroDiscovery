# T477_fc_threshold_eval_comparison: FC Threshold Sensitivity Comparison
## Task Description

Compare model sensitivity to FC edge threshold (5/10/20%) across models; identify threshold-stable models.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Threshold protocol identical to T425.

- Stability score per model.

- Save artefacts under `models/benchmark_results/T477_fc_threshold_eval_comparison/`.


## Expected Output

Expected output artifact(s):

- `threshold_sensitivity.csv`

- `threshold_stability_rank.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
