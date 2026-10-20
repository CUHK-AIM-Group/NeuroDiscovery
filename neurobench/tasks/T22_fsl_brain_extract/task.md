# Benchmark Test Case 22: FSL Brain Extraction (BET)

## Task Description

Load local T1w and run BET brain extraction.

## Input Requirement

Required input(s):

- Local T1w image (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T24_fsl_brain_extract/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- brain.nii.gz
- brain_mask.nii.gz

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
