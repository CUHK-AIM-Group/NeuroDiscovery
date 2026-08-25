# T428_optimizer_comparison: Optimizer Comparison: Adam vs. AdamW vs. SGD
## Task Description

Compare optimizers (Adam / AdamW / SGD+momentum with cosine schedule) for a GCN on both settings; learning rates re-tuned per optimizer on val only.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- LR tuning protocol documented (val-only).

- Save artefacts under `models/benchmark_results/T428_optimizer_comparison/`.


## Expected Output

Expected output artifact(s):

- `optimizer_metrics.csv`

- `optimizer_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
