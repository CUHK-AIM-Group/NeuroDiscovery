# T493_flops_estimation_report: FLOPs Estimation Report
## Task Description

Estimate FLOPs per forward pass per model (thop/fvcore or manual accounting); relate FLOPs to performance.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Method of FLOP counting documented.

- FLOPs-vs-performance scatter included.

- Save artefacts under `models/benchmark_results/T493_flops_estimation_report/`.


## Expected Output

Expected output artifact(s):

- `flops_table.csv`

- `flops_perf.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
