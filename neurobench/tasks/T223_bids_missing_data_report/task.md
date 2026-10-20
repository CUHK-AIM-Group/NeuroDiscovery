# T223_bids_missing_data_report: BIDS Missing-Data Report
## Task Description

Build a subject x modality x session completeness matrix for a BIDS dataset and report missing/expected files per the study protocol.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Expected protocol defined in a small config file kept with output.

- Report both missing files and unexpected extras.

- Save all generated artifacts to:
  - benchmark_results/T223_bids_missing_data_report/


## Expected Output

Expected output artifact(s):

- `completeness_matrix.csv`

- `missing_data_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
