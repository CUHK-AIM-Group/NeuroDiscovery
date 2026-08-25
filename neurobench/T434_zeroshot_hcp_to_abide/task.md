# T434_zeroshot_hcp_to_abide: Zero-Shot Cross-Dataset Transfer
## Task Description

Evaluate a model trained on HCP directly on ABIDE without any finetuning (common atlas); quantify the transfer gap vs. in-domain training.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Common atlas mandatory.

- Report both in-domain and transfer numbers.

- Save artefacts under `models/benchmark_results/T434_zeroshot_hcp_to_abide/`.


## Expected Output

Expected output artifact(s):

- Transfer comparison CSV

- `transfer_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
