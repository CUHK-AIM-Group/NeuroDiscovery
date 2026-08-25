# T469_importance_fold_stability: Importance Stability Across Folds
## Task Description

Measure stability of ROI importance rankings across folds per model: pairwise rank correlation between fold-specific rankings.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Per-fold importance from matched checkpoints.

- Stability ranking across models.

- Save artefacts under `models/benchmark_results/T469_importance_fold_stability/`.


## Expected Output

Expected output artifact(s):

- `fold_stability.csv`

- `stability_heatmap.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
