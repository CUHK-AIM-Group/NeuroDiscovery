# T234_anonymization_verification: Anonymization Verification Report
## Task Description

Verify a dataset is de-identified: scan DICOM/JSON/NIfTI headers and filenames for residual PHI (names, dates beyond year, MRN patterns).

## Input Requirement

Required input(s):

- Dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Pattern list (PHI fields + regexes) kept with the output.

- Findings graded critical/minor; no auto-fixing.

- Save all generated artifacts to:
  - benchmark_results/T234_anonymization_verification/


## Expected Output

Expected output artifact(s):

- `phi_scan_report.csv`

- `remediation_plan.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
