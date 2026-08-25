# T181_nilearn_group_ica_canica: Nilearn Group ICA (CanICA) Pipeline
## Task Description

Run CanICA group ICA over a set of resting-state subjects, extract the 20-component atlas, and back-reconstruct subject maps for downstream analysis.

## Input Requirement

Required input(s):

- List of preprocessed resting-state 4D NIfTIs (>= 5 subjects, required)


If any required input is missing, return:

- Missing required input


## Constraints

- `nilearn.decomposition.CanICA` with 20 components, standard smoothing 6 mm.

- Label components against a reference atlas (Yeo 7-network overlap table).

- Save all generated artifacts to:
  - benchmark_results/T181_nilearn_group_ica_canica/


## Expected Output

Expected output artifact(s):

- `canica_components.nii.gz` + component montage PNG

- `yeo_overlap_table.csv`

- Per-subject back-reconstructed maps


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
