# T230_dicom_integrity_check: DICOM Transfer Integrity Check
## Task Description

Verify a transferred DICOM study: instance counts per series vs. the sender manifest, readable headers, and no zero-byte files.

## Input Requirement

Required input(s):

- DICOM directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Compare against the sender manifest when provided; otherwise use internal consistency (SeriesNumber contiguous).

- Save all generated artifacts to:
  - benchmark_results/T230_dicom_integrity_check/


## Expected Output

Expected output artifact(s):

- `integrity_report.csv`

- `corrupted_instances.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
