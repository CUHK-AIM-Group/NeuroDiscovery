# T472_disagreement_case_mining: Model Disagreement Case Mining
## Task Description

Mine subjects where top models disagree: cluster the disagreement cases, characterize them (site, motion, age), and write case summaries.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Disagreement defined on hard labels.

- At least 10 case summaries.

- Save artefacts under `models/benchmark_results/T472_disagreement_case_mining/`.


## Expected Output

Expected output artifact(s):

- `disagreement_cases.csv`

- `case_summaries.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
