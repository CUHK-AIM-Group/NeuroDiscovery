# T183_cat12_vbm_pipeline: CAT12 VBM Pipeline
## Task Description

Run a CAT12 VBM analysis for a group of T1w images: segmentation, DARTEL normalization, modulation, smoothing, and a two-group GMV comparison.

## Input Requirement

Required input(s):

- T1w NIfTIs for both groups (>= 10 per group, required)

- Group assignment file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- CAT12 via SPM batch (document CAT12 release).

- Smoothing 8 mm FWHM; TIV as covariate; threshold with TFCE or FWE correction.

- Save all generated artifacts to:
  - benchmark_results/T183_cat12_vbm_pipeline/


## Expected Output

Expected output artifact(s):

- Modulated normalized GM maps

- `group_diff_tstat.nii` + rendered PNG

- `cat12_qc_report.pdf`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
