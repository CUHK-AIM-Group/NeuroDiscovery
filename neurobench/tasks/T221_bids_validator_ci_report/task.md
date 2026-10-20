# T221_bids_validator_ci_report: BIDS Validator CI-Style Report
## Task Description

Run bids-validator across a dataset and produce a CI-style report suitable for gating merges: errors/warnings summarized with per-file detail.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Exit code semantics documented (0 = pass).

- Distinguish errors from warnings; do not silently ignore codes.

- Save all generated artifacts to:
  - benchmark_results/T221_bids_validator_ci_report/


## Expected Output

Expected output artifact(s):

- `validation_summary.json`

- `validation_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
