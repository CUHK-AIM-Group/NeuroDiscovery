# T429_early_stopping_sweep: Early-Stopping Patience Sweep
## Task Description

Sweep early-stopping patience (5/10/20 epochs) for a GNN; report effect on final metric and training time.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Patience is the only change.

- Save artefacts under `models/benchmark_results/T429_early_stopping_sweep/`.


## Expected Output

Expected output artifact(s):

- `patience_metrics.csv`

- `patience_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
