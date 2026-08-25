# T229_dicom_series_selection_report: DICOM Series Selection Report
## Task Description

For a study, classify each DICOM series (T1w, T2w, BOLD task/rest, DWI, fmap, ASL, localizer, other) from header metadata and propose the BIDS mapping.

## Input Requirement

Required input(s):

- DICOM directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Classification rules kept as a readable config.

- Ambiguous series flagged for human review, not guessed.

- Save all generated artifacts to:
  - benchmark_results/T229_dicom_series_selection_report/


## Expected Output

Expected output artifact(s):

- `series_classification.csv`

- `proposed_bids_mapping.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
