# T438_test_retest_icc: Test-Retest ICC of Predictions
## Task Description

Train on HCP session-1 resting-state and predict on session-2 (retest): compute ICC of per-subject predictions across sessions.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Subject pairing verified from IDs.

- ICC(2,1) reported with 95% CI.

- Save artefacts under `models/benchmark_results/T438_test_retest_icc/`.


## Expected Output

Expected output artifact(s):

- Metrics CSV per condition

- `robustness_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
