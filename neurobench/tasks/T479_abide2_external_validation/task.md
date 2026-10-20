# T479_abide2_external_validation: External Validation on ABIDE-II
## Task Description

Validate ABIDE-I-trained models on ABIDE-II (no finetuning): metrics, degradation vs. in-domain, and per-site breakdown.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Common atlas across ABIDE-I/II.

- No finetuning; document any preprocessing alignment.

- Save artefacts under `models/benchmark_results/T479_abide2_external_validation/`.


## Expected Output

Expected output artifact(s):

- `abide2_metrics.csv`

- `external_validity_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
