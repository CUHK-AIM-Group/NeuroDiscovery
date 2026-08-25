# T426_class_imbalance_strategies: Class Imbalance: Weighting vs. Oversampling
## Task Description

Compare class-imbalance handling for ABIDE dx: weighted loss vs. random oversampling vs. none, with the same GCN.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Report balanced accuracy + F1 in addition to accuracy.

- Save artefacts under `models/benchmark_results/T426_class_imbalance_strategies/`.


## Expected Output

Expected output artifact(s):

- `imbalance_comparison.csv`

- `imbalance_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
