# T490_training_time_pareto: Training-Time vs. Performance Pareto
## Task Description

Plot the training-time vs. performance Pareto front across models per setting; identify Pareto-optimal models.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Wall time from training logs (same hardware).

- Pareto front marked on the plot.

- Save artefacts under `models/benchmark_results/T490_training_time_pareto/`.


## Expected Output

Expected output artifact(s):

- `pareto.csv`

- `pareto_front.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
