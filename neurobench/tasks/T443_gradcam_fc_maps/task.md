# T443_gradcam_fc_maps: Grad-CAM-style Saliency on FC-CNN
## Task Description

Train the 2D-CNN-on-FC model and produce Grad-CAM-style saliency maps; aggregate per-ROI importance across subjects.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Saliency method equations documented.

- Per-ROI aggregation as mean |saliency|.

- Save artefacts under `models/benchmark_results/T443_gradcam_fc_maps/`.


## Expected Output

Expected output artifact(s):

- Per-ROI importance CSV

- Representative saliency/attribution PNG

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
