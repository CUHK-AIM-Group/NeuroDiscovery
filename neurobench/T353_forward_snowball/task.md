# T353_forward_snowball: Forward Snowball Sampling
## Task Description

From a seed set of papers, perform forward snowballing: collect citing papers via OpenAlex/Semantic Scholar, screen, and iterate one level.

## Input Requirement

Required input(s):

- Seed paper list or query (required)

- Screening criteria where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Citing-paper API source documented.

- One iteration level; counts per stage reported.

- Save all generated artifacts to:
  - benchmark_results/T353_forward_snowball/


## Expected Output

Expected output artifact(s):

- `forward_results.csv`

- `stage_counts.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
