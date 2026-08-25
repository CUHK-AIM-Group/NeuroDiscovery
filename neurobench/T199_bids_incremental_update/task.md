# T199_bids_incremental_update: BIDS Incremental Update Pipeline
## Task Description

Add newly acquired sessions to an existing BIDS dataset: convert, merge, re-validate, update participants/scans TSVs, and emit a changelog diff of what changed.

## Input Requirement

Required input(s):

- Existing BIDS dataset (required)

- New DICOM/NIfTI session directories (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Existing subjects/sessions must remain byte-identical.

- Changelog lists added/modified files only.

- Save all generated artifacts to:
  - benchmark_results/T199_bids_incremental_update/


## Expected Output

Expected output artifact(s):

- Updated BIDS dataset

- `incremental_changelog.md`

- Fresh `bids_validation_report.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
