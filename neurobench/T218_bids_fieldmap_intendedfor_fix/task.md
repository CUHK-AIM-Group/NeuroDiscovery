# T218_bids_fieldmap_intendedfor_fix: BIDS Fieldmap IntendedFor Repair
## Task Description

Audit and repair `IntendedFor` links in a BIDS dataset: every fieldmap must point to existing BOLD/DWI files, and every BOLD needing SDC must have an assigned fieldmap.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Report broken/dangling links before fixing.

- Do not modify raw NIfTI data; JSON sidecars only.

- Save all generated artifacts to:
  - benchmark_results/T218_bids_fieldmap_intendedfor_fix/


## Expected Output

Expected output artifact(s):

- `intendedfor_audit.csv` (before/after)

- Fixed JSON sidecars

- `unmatched_bold_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
