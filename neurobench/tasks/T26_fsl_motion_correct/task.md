# Benchmark Test Case 26: FSL Motion Correction (MCFLIRT)

## Task Description

Load local bold.nii.gz and run MCFLIRT motion correction.

## Input Requirement

Required input(s):

- bold.nii.gz (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T28_fsl_motion_correct/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Motion-corrected 4D BOLD
- Motion parameter file

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
