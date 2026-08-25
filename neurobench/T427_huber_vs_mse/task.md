# T427_huber_vs_mse: Loss Comparison: Huber vs. MSE (Regression)
## Task Description

Compare Huber vs. MSE loss for HCP age regression with the same model; Huber delta documented.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Delta=1.0 unless justified.

- Report outlier sensitivity (worst-10-subject MAE).

- Save artefacts under `models/benchmark_results/T427_huber_vs_mse/`.


## Expected Output

Expected output artifact(s):

- `loss_comparison.csv`

- `outlier_analysis.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
