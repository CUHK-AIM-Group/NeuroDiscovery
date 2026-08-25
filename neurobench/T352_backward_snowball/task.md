# T352_backward_snowball: Backward Snowball Sampling
## Task Description

From a seed set of papers, perform backward snowballing: collect all references, screen by title/abstract criteria, and iterate one level.

## Input Requirement

Required input(s):

- Seed paper list or query (required)

- Screening criteria where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Screening criteria file required as input.

- One iteration level; counts per stage reported.

- Save all generated artifacts to:
  - benchmark_results/T352_backward_snowball/


## Expected Output

Expected output artifact(s):

- `snowball_results.csv`

- `stage_counts.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
