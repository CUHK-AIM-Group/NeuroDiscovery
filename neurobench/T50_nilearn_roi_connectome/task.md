# Benchmark Test Case 50: Nilearn ROI Connectome

## Task Description

Load local ROI time series and compute ROI-to-ROI functional connectivity matrix.

## Input Requirement

Required input(s):

- ROI time series file (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use Nilearn-compatible workflow and functions.
- Save all generated artifacts to:
  - benchmark_results/T73_nilearn_roi_connectome/

## Expected Output

Expected output artifact(s):

- connectome.npy (R x R)
- connectome.csv (R x R)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
