# T466_icv_confound_check: ICV / Head-Size Confound Check
## Task Description

Check whether model predictions correlate with intracranial volume (a proxy confound): partial correlations controlling for the label.

## Input Requirement

Required input(s):

- Per-model 5-fold test predictions/metrics from the benchmark runs (required)

- Subject list + labels + site/sex/age covariates CSV (required)


If any required input is missing, return:

- Missing required input


## Constraints

- ICV from FreeSurfer eTIV.

- Report partial r + p per model.

- Save artefacts under `models/benchmark_results/T466_icv_confound_check/`.


## Expected Output

Expected output artifact(s):

- `icv_confound.csv`

- `confound_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
