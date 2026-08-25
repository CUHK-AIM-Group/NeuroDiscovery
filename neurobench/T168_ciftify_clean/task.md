# T168_ciftify_clean: ciftify Clean + Dense Connectome
## Task Description

From ciftify fMRI output, run `ciftify_clean_img` denoising and compute a dense functional connectome with `cifti_conn_matrix`.

## Input Requirement

Required input(s):

- ciftify fMRI output for one subject (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Cleaning: detrend, band-pass 0.01-0.1 Hz, 24-motion + WM/CSF regression; document the cleaning config.

- Smoothing FWHM must be stated.

- Save all generated artifacts to:
  - benchmark_results/T168_ciftify_clean/


## Expected Output

Expected output artifact(s):

- `cleaned.dtseries.nii`

- `dconn.nii` or dense correlation matrix

- `clean_config.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
