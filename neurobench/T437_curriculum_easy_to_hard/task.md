# T437_curriculum_easy_to_hard: Curriculum Learning
## Task Description

Train with a curriculum: easy subjects (high tSNR / low motion) first, progressively adding harder ones; compare against shuffled training.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Difficulty score defined from QC metrics and documented.

- Save artefacts under `models/benchmark_results/T437_curriculum_easy_to_hard/`.


## Expected Output

Expected output artifact(s):

- Transfer comparison CSV

- `transfer_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
