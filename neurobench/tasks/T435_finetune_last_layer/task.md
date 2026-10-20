# T435_finetune_last_layer: Finetune Last Layer Only
## Task Description

Transfer with only the readout layer trainable (backbone frozen); compare against full finetuning from T432.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Only readout parameters update (verify via optimizer param groups).

- Save artefacts under `models/benchmark_results/T435_finetune_last_layer/`.


## Expected Output

Expected output artifact(s):

- Transfer comparison CSV

- `transfer_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
