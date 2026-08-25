# T432_pretrain_hcp_finetune_abide: Pretrain HCP -> Finetune ABIDE
## Task Description

Pretrain a GNN on HCP (age regression), finetune on ABIDE dx; compare against training from scratch on ABIDE only.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Same atlas for both datasets.

- Finetune protocol (frozen layers, LR) documented.

- Save artefacts under `models/benchmark_results/T432_pretrain_hcp_finetune_abide/`.


## Expected Output

Expected output artifact(s):

- Transfer comparison CSV

- `transfer_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
