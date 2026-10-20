# T431_edge_dropout_augmentation: Augmentation: Edge Dropout
## Task Description

Evaluate edge-dropout augmentation (drop 10% edges per epoch) for a GCN on both settings; compare against no augmentation.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Dropout applied to training graphs only.

- Save artefacts under `models/benchmark_results/T431_edge_dropout_augmentation/`.


## Expected Output

Expected output artifact(s):

- `augmentation_metrics.csv`

- `augmentation_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
