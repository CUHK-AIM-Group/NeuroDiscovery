# T483_zeroshot_crossdataset_report: Zero-Shot Cross-Dataset Report
## Task Description

Consolidate zero-shot transfer results (HCP->ABIDE, ABIDE-I->II) into one cross-dataset generalization report with a shift-difficulty ranking.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Metrics pulled from the corresponding task outputs.

- Ranking justified by shift metrics (e.g. MMD).

- Save artefacts under `models/benchmark_results/T483_zeroshot_crossdataset_report/`.


## Expected Output

Expected output artifact(s):

- `crossdataset_report.md`

- `zeroshot_matrix.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
