# T198_onboarding_dicom_to_bids_qc: DICOM-to-BIDS Onboarding Pipeline
## Task Description

Full onboarding chain for a new scanner shipment: DICOM sorting, dcm2bids conversion, bids-validator, MRIQC, and a final onboarding report with pass/fail.

## Input Requirement

Required input(s):

- DICOM tar/directory as delivered by the scanner (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Every stage logs its tool + version.

- Any stage failure must quarantine the subject and continue with the rest.

- Save all generated artifacts to:
  - benchmark_results/T198_onboarding_dicom_to_bids_qc/


## Expected Output

Expected output artifact(s):

- BIDS tree + validation report

- MRIQC outputs

- `onboarding_report.md` (per-subject status table)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
