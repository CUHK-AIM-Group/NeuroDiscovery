# T500_evaluation_protocol_document: Evaluation Protocol Document
## Task Description

Write the evaluation protocol document: datasets, splits, seeds, metrics, statistical tests, and reporting rules, so a new model can be evaluated identically.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Versioned document; every number traceable to a task output.

- Includes the exact commands to run a new model.

- Save artefacts under `models/benchmark_results/T500_evaluation_protocol_document/`.


## Expected Output

Expected output artifact(s):

- `EVALUATION_PROTOCOL.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
