# Benchmark Test Case 49: Nilearn ROI Time Series Extraction

## Task Description

Load local preprocessed BOLD + confounds + atlas/parcellation, extract ROI time series, and output roi_timeseries.csv (T x R).

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
  - benchmark_results/T72_nilearn_roi_ts/

## Expected Output

Expected output artifact(s):

- roi_timeseries.csv (T x R)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
