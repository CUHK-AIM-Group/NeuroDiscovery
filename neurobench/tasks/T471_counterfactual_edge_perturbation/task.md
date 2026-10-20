# T471_counterfactual_edge_perturbation: Counterfactual Edge Perturbation
## Task Description

Perturbation analysis: remove top-k important edges (per importance method) and measure prediction change; validate that importance rankings are causal-ish.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- k grid documented.

- Compared against random-edge removal baseline.

- Save artefacts under `models/benchmark_results/T471_counterfactual_edge_perturbation/`.


## Expected Output

Expected output artifact(s):

- `perturbation_curves.csv`

- `perturbation_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
