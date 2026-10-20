# T482_pretrain_gain_analysis: Pretraining Gain Analysis
## Task Description

Quantify the gain of pretraining (SSL or HCP) across models: pretrained vs. scratch comparison table with significance.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Paired comparisons per model.

- Gain attributed to pretraining only (same budget).

- Save artefacts under `models/benchmark_results/T482_pretrain_gain_analysis/`.


## Expected Output

Expected output artifact(s):

- `pretrain_gain.csv`

- `pretrain_gain.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
