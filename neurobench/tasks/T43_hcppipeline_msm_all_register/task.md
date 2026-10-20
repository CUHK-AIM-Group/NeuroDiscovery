# Benchmark Test Case 43: HCP MSMAll Surface Registration

## Task Description

Load local structural + functional surface data and run MSMAll registration.

## Input Requirement

Required input(s):

- Structural surface data (required)
- Functional surface data (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T58_hcp_msm_all_register/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- MSMAll surface registration outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
