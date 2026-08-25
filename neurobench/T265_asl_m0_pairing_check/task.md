# T265_asl_m0_pairing_check: ASL M0 Pairing Check
## Task Description

Verify every ASL run has an associated M0 scan (separate file or embedded), per the ASL-BIDS specification.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Follow the ASL-BIDS M0 pairing rules.

- Report pairing type per run (separate/included/absent).

- Save all generated artifacts to:
  - benchmark_results/T265_asl_m0_pairing_check/


## Expected Output

Expected output artifact(s):

- `asl_m0_pairing.csv`

- `asl_bids_findings.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
