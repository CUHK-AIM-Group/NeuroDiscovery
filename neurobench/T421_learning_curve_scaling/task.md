# T421_learning_curve_scaling: Learning Curve: Data Scaling
## Task Description

Measure the learning curve of BrainGNN: train on 10/25/50/75/100% of the training data (stratified subsets, fixed test), plot performance vs. data fraction.

## Input Requirement

Required input(s):

- ROI-level FC matrices per subject (NPZ), built from a chosen atlas (e.g. `schaefer_200_7net`, `aal_116`)

- Subject list file (`ready_subjects.txt`)

- Labels CSV (HCP age: `data/hcp_age_labels.csv`; ABIDE dx: `data/abide_dx_labels.csv`)

- Atlas name and ROI count (must match FC dimension)


If any required input is missing, return:

- Missing required input


## Constraints

- Subsets nested (10% is a subset of 25% etc.), seed fixed.

- Save artefacts under `models/benchmark_results/T421_learning_curve_scaling/`.


## Expected Output

Expected output artifact(s):

- `learning_curve.csv`

- `learning_curve.png`

- `result_YYYYMMDD_HHMMSS.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- Curve monotonicity discussed; saturation point estimated.

- This test case is manually evaluated.
