# Benchmark Test Case 35: HCP Structural FreeSurfer Stage

## Task Description

Load local PreFreeSurfer outputs and run HCP FreeSurfer reconstruction stage.

## Input Requirement

Required input(s):

- PreFreeSurfer output directory (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T50_hcp_structural_free_surfer/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Complete FreeSurfer reconstruction outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
