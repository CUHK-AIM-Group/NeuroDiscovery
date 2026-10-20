# T464_fairness_gap_summary: Fairness Gap Summary
## Task Description

Consolidate subgroup gaps (sex, age, site, motion) into one fairness scorecard per model with an overall fairness ranking.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Scorecard schema fixed.

- Ranking rule documented.

- Save artefacts under `models/benchmark_results/T464_fairness_gap_summary/`.


## Expected Output

Expected output artifact(s):

- `fairness_scorecard.csv`

- `fairness_summary.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
