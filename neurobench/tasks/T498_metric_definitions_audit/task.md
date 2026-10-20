# T498_metric_definitions_audit: Metric Definitions Audit
## Task Description

Audit every reported metric: definition, averaging mode (macro/weighted/micro), implementation source, and edge-case behavior; publish the metric glossary.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Each metric linked to its code location.

- Inconsistencies found are listed with impact.

- Save artefacts under `models/benchmark_results/T498_metric_definitions_audit/`.


## Expected Output

Expected output artifact(s):

- `metric_glossary.md`

- `metric_audit.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
