# T224_bids_duplicates_detection: BIDS Duplicate File Detection
## Task Description

Detect duplicate content in a BIDS dataset: same checksum under different paths, near-duplicate NIfTIs (same hash after gzip normalization), and duplicated sessions.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Normalize .nii.gz before hashing (gunzip stream) so re-compressed duplicates are caught.

- Never delete; produce a quarantine list only.

- Save all generated artifacts to:
  - benchmark_results/T224_bids_duplicates_detection/


## Expected Output

Expected output artifact(s):

- `duplicates.csv` (path pairs + hash)

- `duplicate_resolution_plan.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
