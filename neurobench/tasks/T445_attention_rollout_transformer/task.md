# T445_attention_rollout_transformer: Attention Rollout for TS-Transformer
## Task Description

Apply attention rollout to the trained ROI time-series transformer; visualize ROI-to-ROI attention flow for example subjects.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Rollout implementation cited.

- Examples span both settings.

- Save artefacts under `models/benchmark_results/T445_attention_rollout_transformer/`.


## Expected Output

Expected output artifact(s):

- Per-ROI importance CSV

- Representative saliency/attribution PNG

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
