# T125_oasis_bids_staging: OASIS-3 Download and BIDS Staging
## Task Description

Download an OASIS-3 imaging subset (T1w + T2w for a given subject list) and
stage it into a BIDS-compliant dataset directory.

## Input Requirement

Required input(s):

- OASIS-3 subject/session list file (required)
- OASIS-3 access credentials or pre-downloaded archive path (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Organize output as `sub-<label>/ses-<label>/anat/` with BIDS filenames.
- Generate `dataset_description.json` and `participants.tsv`.
- The staged dataset must pass `bids-validator` with no errors.
- Save all generated artifacts to:
  - benchmark_results/T125_oasis_bids_staging/

## Expected Output

Expected output artifact(s):

- BIDS dataset directory tree
- `bids_validation_report.txt` (validator output)
- `data_inventory.csv` (one row per staged file with size + checksum)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
