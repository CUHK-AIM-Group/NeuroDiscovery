# T462_age_group_metrics: Age-Group Metrics
## Task Description

Report performance per age group (tertiles or study-defined bands) for every model; test for age-related performance gradients.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Bands identical across models.

- Gradient tested with trend test.

- Save artefacts under `models/benchmark_results/T462_age_group_metrics/`.


## Expected Output

Expected output artifact(s):

- `age_group_metrics.csv`

- `age_gradient.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
