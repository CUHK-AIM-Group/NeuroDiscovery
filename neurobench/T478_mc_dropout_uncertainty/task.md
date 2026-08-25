# T478_mc_dropout_uncertainty: MC-Dropout Uncertainty Evaluation
## Task Description

Estimate predictive uncertainty via MC dropout for each GNN; evaluate uncertainty quality (error-vs-uncertainty correlation, selective prediction AUC).

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- T=30 stochastic forward passes.

- Selective-prediction curves per model.

- Save artefacts under `models/benchmark_results/T478_mc_dropout_uncertainty/`.


## Expected Output

Expected output artifact(s):

- `uncertainty_metrics.csv`

- `selective_curves.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
