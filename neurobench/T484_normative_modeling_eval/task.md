# T484_normative_modeling_eval: Normative Modeling Evaluation
## Task Description

Fit a normative model (PCN-style) on controls and evaluate deviation scores (z-scores) for patients; compare separation against direct classifiers.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Normative model fit on controls only.

- Report AUC of deviation-score classification.

- Save artefacts under `models/benchmark_results/T484_normative_modeling_eval/`.


## Expected Output

Expected output artifact(s):

- `normative_metrics.csv`

- `deviation_maps.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
