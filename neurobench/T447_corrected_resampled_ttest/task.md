# T447_corrected_resampled_ttest: Corrected Resampled t-Test Between Models
## Task Description

Compare model pairs with Nadeau-Bengio corrected resampled t-test over the 5-fold results; produce a pairwise p-value matrix.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Correction factor applied (test/train ratio).

- Multiple-comparison note included.

- Save artefacts under `models/benchmark_results/T447_corrected_resampled_ttest/`.


## Expected Output

Expected output artifact(s):

- `pairwise_pvalues.csv`

- `ttest_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
