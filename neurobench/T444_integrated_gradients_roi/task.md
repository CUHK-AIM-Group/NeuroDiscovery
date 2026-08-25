# T444_integrated_gradients_roi: Integrated Gradients ROI Importance
## Task Description

Compute Integrated Gradients attributions for a trained GNN; export per-ROI importance rankings per fold.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Baseline (zero graph) documented.

- Attributions averaged per class for classification.

- Save artefacts under `models/benchmark_results/T444_integrated_gradients_roi/`.


## Expected Output

Expected output artifact(s):

- Per-ROI importance CSV

- Representative saliency/attribution PNG

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
