# T433_ssl_linear_probe: SSL Pretrain + Linear Probe
## Task Description

Use the T407 contrastive-pretrained encoder (or train one here), freeze it, and train linear probes for both settings; report vs. end-to-end.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Encoder frozen during probing.

- Probe protocol identical across settings.

- Save artefacts under `models/benchmark_results/T433_ssl_linear_probe/`.


## Expected Output

Expected output artifact(s):

- Transfer comparison CSV

- `transfer_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
