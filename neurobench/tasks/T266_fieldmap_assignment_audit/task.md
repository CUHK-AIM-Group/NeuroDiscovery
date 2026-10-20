# T266_fieldmap_assignment_audit: Fieldmap Assignment Audit
## Task Description

Audit fieldmap-to-scan assignment: correct `IntendedFor` coverage, PE-direction pairing sanity (AP/PA), and per-scan SDC readiness.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- PE direction read from JSON sidecars.

- Scans lacking SDC options listed explicitly.

- Save all generated artifacts to:
  - benchmark_results/T266_fieldmap_assignment_audit/


## Expected Output

Expected output artifact(s):

- `fmap_assignment.csv`

- `sdc_readiness_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
