# T217_dcm2bids_multi_session: dcm2bids Multi-Session Batch
## Task Description

Batch-convert multiple subjects x sessions with dcm2bids driven by a participant/session manifest, with per-session success tracking.

## Input Requirement

Required input(s):

- Manifest CSV (subject, session, dicom_dir) (required)

- dcm2bids config JSON (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Per-session logs; failures quarantined, batch continues.

- Final summary distinguishes converted / failed / skipped.

- Save all generated artifacts to:
  - benchmark_results/T217_dcm2bids_multi_session/


## Expected Output

Expected output artifact(s):

- BIDS tree

- `batch_summary.csv`

- Per-session logs directory


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
