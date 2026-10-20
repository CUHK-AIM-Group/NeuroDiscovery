# T439_noise_injection_training: Noise-Robust Training
## Task Description

Train with Gaussian noise augmentation on FC; evaluate clean vs. noisy test performance vs. a model trained without augmentation.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Noise sigma documented; test noise levels grid reported.

- Save artefacts under `models/benchmark_results/T439_noise_injection_training/`.


## Expected Output

Expected output artifact(s):

- Metrics CSV per condition

- `robustness_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
