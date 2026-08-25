# T278_container_size_audit: Container Size Audit with dive
## Task Description

Audit a large analysis image with `dive`: identify the biggest layers, wasted space, and produce a slimming plan.

## Input Requirement


- No interactive input.


## Constraints

- Report per-layer sizes.

- Plan must keep functionality; estimate savings per suggestion.

- Save all generated artifacts to:
  - benchmark_results/T278_container_size_audit/


## Expected Output

Expected output artifact(s):

- `dive_report.txt`

- `slimming_plan.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
