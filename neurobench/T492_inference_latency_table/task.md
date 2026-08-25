# T492_inference_latency_table: Inference Latency Table
## Task Description

Measure per-subject inference latency for every model (batch=1, CPU and GPU); produce a latency table for deployment discussion.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Warmup runs excluded; 100 repetitions.

- Hardware spec recorded.

- Save artefacts under `models/benchmark_results/T492_inference_latency_table/`.


## Expected Output

Expected output artifact(s):

- `latency_table.csv`

- `latency_notes.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
