# T451_delong_roc_comparison: DeLong ROC Comparison
## Task Description

Compare classification AUCs between model pairs with the DeLong test on pooled test predictions.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Same test subjects across compared models (verify).

- Report AUC differences + CIs, not just p-values.

- Save artefacts under `models/benchmark_results/T451_delong_roc_comparison/`.


## Expected Output

Expected output artifact(s):

- `delong_results.csv`

- `roc_overlays.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
