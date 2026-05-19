# Benchmark Test Case 40: HCP Diffusion Preprocessing

## Task Description

Load local dwi.nii.gz + bvecs + bvals + AP/PA b0 and run topup+eddy correction.

## Input Requirement

Required input(s):

- dwi.nii.gz (required)
- bvecs (required)
- bvals (required)
- AP/PA b0 data (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T55_hcp_diffusion_preproc/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Corrected DWI outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
