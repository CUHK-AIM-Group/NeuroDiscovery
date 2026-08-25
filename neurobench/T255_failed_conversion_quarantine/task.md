# T255_failed_conversion_quarantine: Failed Conversion Quarantine
## Task Description

Collect failed conversions from pipeline logs, move (copy + plan) the affected sourcedata into a quarantine area, and produce a retry list.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Copy, never move originals; emit a quarantine plan instead of acting if destructive.

- Retry list grouped by failure reason.

- Save all generated artifacts to:
  - benchmark_results/T255_failed_conversion_quarantine/


## Expected Output

Expected output artifact(s):

- `quarantine/` directory or plan

- `retry_list.csv`

- `failure_reasons.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
