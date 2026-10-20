# T491_model_size_vs_perf: Model Size vs. Performance
## Task Description

Compare parameter count vs. performance across models; compute parameters-per-point efficiency.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Parameter counts from model summaries (verified).

- Log-scale plot.

- Save artefacts under `models/benchmark_results/T491_model_size_vs_perf/`.


## Expected Output

Expected output artifact(s):

- `size_vs_perf.csv`

- `size_vs_perf.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
