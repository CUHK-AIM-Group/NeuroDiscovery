# Benchmark Test Case 25: FSL Subcortical Segmentation (FIRST)

## Task Description

Load local brain.nii.gz and run FIRST for fine subcortical segmentation.

## Input Requirement

Required input(s):

- brain.nii.gz (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T27_fsl_subcort_seg/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Subcortical segmentation labels (aseg-style)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
