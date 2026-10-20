# T320_smoke_suite_bidsapps: Smoke Suite for BIDS Apps
## Task Description

A single smoke-test script that verifies all containerized BIDS apps used by the lab start and respond (--version/--help) with expected versions.

## Input Requirement


- No interactive input.


## Constraints

- Expected versions in a config file.

- One command runs all checks.

- Save all generated artifacts to:
  - benchmark_results/T320_smoke_suite_bidsapps/


## Expected Output

Expected output artifact(s):

- `smoke_bidsapps.sh`

- `expected_versions.yaml`

- `smoke_report.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
