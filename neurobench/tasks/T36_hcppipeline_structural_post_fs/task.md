# Benchmark Test Case 36: HCP Structural PostFreeSurfer

## Task Description

Load local FreeSurfer outputs and run HCP PostFreeSurfer derivative processing.

## Input Requirement

Required input(s):

- FreeSurfer output directory (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T51_hcp_structural_post_fs/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Final PostFreeSurfer structural outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
