# Benchmark Test Case 53: Nilearn Full ROI Pipeline

## Task Description

Load local preprocessed BOLD + confounds + atlas and run full ROI time-series plus connectivity extraction pipeline.

## Input Requirement

Required input(s):

- Preprocessed BOLD image (required)
- Confounds file (required)
- Atlas/parcellation image (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use Nilearn-compatible workflow and functions.
- Save all generated artifacts to:
  - benchmark_results/T76_nilearn_full_roi_pipeline/

## Expected Output

Expected output artifact(s):

- ROI time-series outputs
- ROI connectivity outputs
- All ROI pipeline artifacts

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
