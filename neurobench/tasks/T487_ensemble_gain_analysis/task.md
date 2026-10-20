# T487_ensemble_gain_analysis: Ensemble Gain Analysis
## Task Description

Analyze when ensembling helps: ensemble gain vs. member diversity (prediction disagreement), across settings.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Diversity via Q-statistic or disagreement measure.

- Scatter: diversity vs. gain.

- Save artefacts under `models/benchmark_results/T487_ensemble_gain_analysis/`.


## Expected Output

Expected output artifact(s):

- `ensemble_gain.csv`

- `diversity_gain.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
