# T254_derivative_completeness_report: Derivative Completeness Report
## Task Description

Verify fMRIPrep/XCP-D output completeness per subject: every expected derivative file present (per a manifest template), missing ones listed.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Expected-file template kept as config.

- Report coverage percent per subject and overall.

- Save all generated artifacts to:
  - benchmark_results/T254_derivative_completeness_report/


## Expected Output

Expected output artifact(s):

- `derivative_completeness.csv`

- `missing_derivatives.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
