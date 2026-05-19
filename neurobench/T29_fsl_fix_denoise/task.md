# Benchmark Test Case 29: FSL FIX Denoising

## Task Description

Load local MELODIC output folder and run FIX automatic denoising.

## Input Requirement

Required input(s):

- MELODIC output folder (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T31_fsl_fix_denoise/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Denoised BOLD output

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
