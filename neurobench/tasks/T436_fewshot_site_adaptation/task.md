# T436_fewshot_site_adaptation: Few-Shot Site Adaptation
## Task Description

Adapt an ABIDE-trained model to a held-out site with k=5 labeled subjects per site; report adaptation gain over zero-shot.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- k fixed at 5; selection of the 5 documented (seeded).

- Save artefacts under `models/benchmark_results/T436_fewshot_site_adaptation/`.


## Expected Output

Expected output artifact(s):

- Transfer comparison CSV

- `transfer_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
