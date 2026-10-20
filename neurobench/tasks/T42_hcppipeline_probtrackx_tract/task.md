# Benchmark Test Case 42: HCP Probtrackx Tractography

## Task Description

Load local bedpostx outputs and run probtrackx fiber tracking.

## Input Requirement

Required input(s):

- Bedpostx output directory (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T57_hcp_probtrackx_tract/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- White matter tractography outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
