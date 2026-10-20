# T496_results_table_latex_export: Results Tables LaTeX Export
## Task Description

Export all evaluation tables (subgroups, robustness, efficiency) to publication-ready LaTeX with consistent formatting and booktabs style.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Bold-best rule documented and consistent.

- Every table compiles standalone.

- Save artefacts under `models/benchmark_results/T496_results_table_latex_export/`.


## Expected Output

Expected output artifact(s):

- `tables/*.tex`

- `tables_preview.pdf`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
