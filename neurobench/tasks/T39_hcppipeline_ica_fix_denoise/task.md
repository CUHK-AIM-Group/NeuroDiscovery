# Benchmark Test Case 39: HCP ICA-FIX Denoising

## Task Description

Load local fMRISurface outputs and run ICA-FIX automatic denoising.

## Input Requirement

Required input(s):

- fMRISurface output directory (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T54_hcp_ica_fix_denoise/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Denoised BOLD outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
