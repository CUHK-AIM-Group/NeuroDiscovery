# T425_fc_threshold_sensitivity: Sensitivity: FC Edge Threshold
## Task Description

Evaluate a GCN's sensitivity to FC binarization threshold (top 5/10/20% edges): same model, same folds, three graphs.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Threshold applied identically across subjects.

- Save artefacts under `models/benchmark_results/T425_fc_threshold_sensitivity/`.


## Expected Output

Expected output artifact(s):

- `threshold_metrics.csv`

- `threshold_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
