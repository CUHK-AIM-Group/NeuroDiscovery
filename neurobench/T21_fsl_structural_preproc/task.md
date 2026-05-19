# Benchmark Test Case 21: FSL Structural Full Preprocessing

## Task Description

Load local T1w (optional T2w/FLAIR), run one-command full structural preprocessing, and output complete fsl_anat folder (brain extraction, tissue segmentation, registration, etc.).

## Input Requirement

Required input(s):

- Local T1w image (required)
- Local T2w image (optional)
- Local FLAIR image (optional)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T23_fsl_structural_preproc/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- fsl_anat/ complete output folder

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
