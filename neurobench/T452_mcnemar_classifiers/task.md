# T452_mcnemar_classifiers: McNemar Test Between Classifiers
## Task Description

Run McNemar's test on paired classification outcomes per model pair; build the discordant-pair tables.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Continuity correction documented.

- Discordant cases exported for error analysis.

- Save artefacts under `models/benchmark_results/T452_mcnemar_classifiers/`.


## Expected Output

Expected output artifact(s):

- `mcnemar_tables.csv`

- `discordant_subjects.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
