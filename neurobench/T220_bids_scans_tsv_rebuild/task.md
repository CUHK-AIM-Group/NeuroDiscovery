# T220_bids_scans_tsv_rebuild: BIDS scans.tsv Rebuild
## Task Description

Rebuild `sub-*/ses-*_scans.tsv` files across a BIDS dataset from the actual files on disk, with acquisition times recovered from JSON sidecars.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Do not invent acquisition times; missing times left as `n/a` and reported.

- Diff against existing scans.tsv must be reported.

- Save all generated artifacts to:
  - benchmark_results/T220_bids_scans_tsv_rebuild/


## Expected Output

Expected output artifact(s):

- Updated `*_scans.tsv` files

- `scans_diff_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
