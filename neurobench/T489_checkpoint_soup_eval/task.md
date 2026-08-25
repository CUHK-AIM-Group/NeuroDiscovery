# T489_checkpoint_soup_eval: Checkpoint Soup Evaluation
## Task Description

Evaluate model soups: average weights of the last-k checkpoints (or greedy soup) per model; compare against best-checkpoint selection.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Soup recipe documented.

- Same eval split as standard runs.

- Save artefacts under `models/benchmark_results/T489_checkpoint_soup_eval/`.


## Expected Output

Expected output artifact(s):

- `soup_metrics.csv`

- `soup_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
