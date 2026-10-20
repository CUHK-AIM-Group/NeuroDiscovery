# T449_demsar_cd_diagram: Demšar Critical-Difference Diagram
## Task Description

Rank all models with the Friedman test and produce a critical-difference diagram (Nemenyi post-hoc) across evaluation settings.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Ranks computed per (dataset x fold).

- CD value shown on the diagram.

- Save artefacts under `models/benchmark_results/T449_demsar_cd_diagram/`.


## Expected Output

Expected output artifact(s):

- `cd_diagram.png`

- `friedman_ranks.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
