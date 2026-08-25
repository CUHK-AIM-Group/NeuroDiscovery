# T424_ablation_no_consist_loss: Ablation: BrainGNN without Consistency Loss
## Task Description

Ablate the consistency loss term from BrainGNN training; quantify its contribution on both settings.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Only the loss term differs; weights of remaining terms unchanged.

- Save artefacts under `models/benchmark_results/T424_ablation_no_consist_loss/`.


## Expected Output

Expected output artifact(s):

- Ablation metrics CSV

- `ablation_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
