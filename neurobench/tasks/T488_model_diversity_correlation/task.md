# T488_model_diversity_correlation: Model Diversity Correlation
## Task Description

Compute pairwise prediction-correlation between models; cluster models into families by prediction similarity.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Correlation on subject-level predictions.

- Hierarchical clustering dendrogram included.

- Save artefacts under `models/benchmark_results/T488_model_diversity_correlation/`.


## Expected Output

Expected output artifact(s):

- `diversity_matrix.csv`

- `model_dendrogram.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
