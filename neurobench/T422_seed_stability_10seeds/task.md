# T422_seed_stability_10seeds: Seed Stability: 10-Seed Run
## Task Description

Run the same model (GCN) with 10 different seeds on fold 0 only; report metric variance attributable to seed choice.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Everything else identical (config, fold, atlas).

- Save artefacts under `models/benchmark_results/T422_seed_stability_10seeds/`.


## Expected Output

Expected output artifact(s):

- `seed_metrics.csv`

- `seed_variance_report.md`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
