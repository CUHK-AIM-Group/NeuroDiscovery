# T140_afni_3ddeconvolve: AFNI 3dDeconvolve Task GLM
## Task Description

Run a single-subject task-fMRI GLM with AFNI `3dDeconvolve` using the provided event timings, producing per-condition beta and t-stat maps.

## Input Requirement

Required input(s):

- Preprocessed task fMRI 4D NIfTI (required)

- Event timing files per condition (AFNI format, required)

- Motion parameter file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `3dDeconvolve` with `-polort A` and motion regressors; censoring above FD 0.5 mm if a censor file is provided.

- Model conditions with the BLOCK or GAM hemodynamic basis and document the choice.

- Save all generated artifacts to:
  - benchmark_results/T140_afni_3ddeconvolve/


## Expected Output

Expected output artifact(s):

- `stats_bucket.nii.gz` (betas + t-stats)

- `design_matrix.xmat` + rendered design matrix PNG

- `fitts` and `errts` time series


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
