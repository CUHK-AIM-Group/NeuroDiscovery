# T480_cobre_external_validation: External Validation on COBRE
## Task Description

Validate schizophrenia-relevant models (or dx-transfer probes) on COBRE data; report transfer metrics honestly.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Label mapping documented (ASD-trained vs. SCZ target).

- If not applicable, run as domain-shift probe and say so.

- Save artefacts under `models/benchmark_results/T480_cobre_external_validation/`.


## Expected Output

Expected output artifact(s):

- `cobre_metrics.csv`

- `cobre_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
