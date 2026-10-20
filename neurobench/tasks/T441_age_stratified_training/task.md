# T441_age_stratified_training: Age-Stratified Training
## Task Description

Train separate models per age band (HCP young vs. older bin) and compare with a pooled model; discuss age-effect confounding.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Bands defined a priori and documented.

- Save artefacts under `models/benchmark_results/T441_age_stratified_training/`.


## Expected Output

Expected output artifact(s):

- Metrics CSV per condition

- `robustness_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
