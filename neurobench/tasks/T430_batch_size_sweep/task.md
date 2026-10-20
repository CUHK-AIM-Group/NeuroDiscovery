# T430_batch_size_sweep: Batch Size Sweep
## Task Description

Sweep batch size (8/16/32/64) for a GNN; report metric + memory + epoch-time trade-offs.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Report peak GPU memory per batch size.

- Save artefacts under `models/benchmark_results/T430_batch_size_sweep/`.


## Expected Output

Expected output artifact(s):

- `batch_metrics.csv`

- `batch_tradeoff.png`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
