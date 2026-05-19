# Benchmark Test Case 28: FSL Resting-state ICA (MELODIC)

## Task Description

Load corrected BOLD and run resting-state ICA with MELODIC.

## Input Requirement

Required input(s):

- Motion-corrected BOLD (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T30_fsl_resting_ica/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- ICA components
- Complete MELODIC report

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
