# T446_shap_tabular_baselines: SHAP for Tabular Baselines
## Task Description

Compute SHAP values for the ridge and random-forest baselines; export global ROI importance and per-subject force summaries.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- SHAP variant (KernelSHAP/TreeSHAP) per model documented.

- Global importance as mean |SHAP|.

- Save artefacts under `models/benchmark_results/T446_shap_tabular_baselines/`.


## Expected Output

Expected output artifact(s):

- Per-ROI importance CSV

- Representative saliency/attribution PNG

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
