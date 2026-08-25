# T476_harmonization_method_comparison: Harmonization Method Comparison
## Task Description

Compare harmonization strategies (none / ComBat / CovBat) by downstream LOSO performance of a fixed model.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Harmonization fit on train only (no leakage).

- Same splits across methods.

- Save artefacts under `models/benchmark_results/T476_harmonization_method_comparison/`.


## Expected Output

Expected output artifact(s):

- `harmonization_comparison.csv`

- `harmonization_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
