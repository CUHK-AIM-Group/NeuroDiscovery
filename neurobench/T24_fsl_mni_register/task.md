# Benchmark Test Case 24: FSL MNI Registration (FLIRT + FNIRT)

## Task Description

Load local brain.nii.gz, run linear and nonlinear registration to MNI152.

## Input Requirement

Required input(s):

- brain.nii.gz (required)
- MNI152 template (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T26_fsl_mni_register/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- FLIRT affine matrix
- FNIRT warp field
- MNI-space registered image

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
