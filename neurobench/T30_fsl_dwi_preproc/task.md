# Benchmark Test Case 30: FSL DWI Preprocessing (TOPUP + EDDY)

## Task Description

Load local dwi.nii.gz + bvecs + bvals + AP/PA b0 and run distortion/eddy correction using TOPUP and EDDY.

## Input Requirement

Required input(s):

- dwi.nii.gz (required)
- bvecs (required)
- bvals (required)
- AP b0 and PA b0 data (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T32_fsl_dwi_preproc/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Corrected DWI output
- Updated bvecs/bvals or equivalent correction artifacts

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
