# Benchmark Test Case 23: FSL Tissue Segmentation (FAST)

## Task Description

Load local brain.nii.gz and run FAST tissue segmentation.

## Input Requirement

Required input(s):

- brain.nii.gz (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T25_fsl_tissue_seg/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- GM probability map
- WM probability map
- CSF probability map
- Bias-corrected output

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
