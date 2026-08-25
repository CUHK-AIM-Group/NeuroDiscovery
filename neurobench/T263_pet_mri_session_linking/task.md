# T263_pet_mri_session_linking: PET-MRI Session Linking
## Task Description

Link PET scans to their anatomically-matching MRI sessions per subject: same session preferred, nearest-date fallback with an interval threshold.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Threshold (days) configurable; fallback links explicitly flagged.

- Unlinkable PET scans reported, not silently dropped.

- Save all generated artifacts to:
  - benchmark_results/T263_pet_mri_session_linking/


## Expected Output

Expected output artifact(s):

- `pet_mri_links.csv`

- `linking_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
