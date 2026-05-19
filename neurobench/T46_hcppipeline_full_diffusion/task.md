# Benchmark Test Case 46: HCP Full Diffusion Pipeline

## Task Description

Load local dwi + bvecs + bvals and run complete HCP diffusion pipeline.

## Input Requirement

Required input(s):

- dwi image (required)
- bvecs (required)
- bvals (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T61_hcp_full_diffusion/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- topup + eddy + bedpostx + probtrackx outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
