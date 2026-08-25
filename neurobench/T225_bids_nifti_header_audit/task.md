# T225_bids_nifti_header_audit: NIfTI Header Audit
## Task Description

Audit NIfTI headers across a BIDS dataset: TR, voxel sizes, qform/sform consistency, dim mismatches between runs, and units fields.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Flag qform != sform cases explicitly.

- Per-modality expected values configurable in a small YAML.

- Save all generated artifacts to:
  - benchmark_results/T225_bids_nifti_header_audit/


## Expected Output

Expected output artifact(s):

- `header_audit.csv`

- `header_anomalies.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
