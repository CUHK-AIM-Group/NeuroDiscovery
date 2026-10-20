# T494_memory_peak_comparison: Peak Memory Comparison
## Task Description

Measure peak training GPU memory per model at the standard batch size; flag models exceeding a 12 GB budget.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Measured via torch.cuda.max_memory_allocated.

- Budget line marked on the plot.

- Save artefacts under `models/benchmark_results/T494_memory_peak_comparison/`.


## Expected Output

Expected output artifact(s):

- `memory_table.csv`

- `memory_budget.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
