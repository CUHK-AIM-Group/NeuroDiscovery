# T228_dicom_header_phi_scrub: DICOM Header PHI Scrub
## Task Description

Scrub protected health information from DICOM headers per a defined whitelist/blacklist profile, and verify no PHI remains.

## Input Requirement

Required input(s):

- DICOM directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use a documented profile (e.g. DICOM PS3.15 basic de-identification).

- Keep a mapping log linking original to scrubbed IDs, stored separately.

- Save all generated artifacts to:
  - benchmark_results/T228_dicom_header_phi_scrub/


## Expected Output

Expected output artifact(s):

- Scrubbed DICOM tree

- `phi_scrub_report.json`

- `verification_scan.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
