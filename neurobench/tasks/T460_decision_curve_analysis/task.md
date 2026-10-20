# T460_decision_curve_analysis: Decision Curve Analysis
## Task Description

Run decision curve analysis for the classifiers: net benefit across threshold probabilities, compared against treat-all/treat-none.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Threshold range justified clinically.

- Curves per model on one figure.

- Save artefacts under `models/benchmark_results/T460_decision_curve_analysis/`.


## Expected Output

Expected output artifact(s):

- `decision_curves.png`

- `net_benefit_table.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
