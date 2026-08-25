# T393_retraction_check: Retraction / EoC Check of References
## Task Description

Check every reference against retraction databases (Crossref retractions, Retraction Watch data): flag retracted or expression-of-concern items.

## Input Requirement

Required input(s):

- Bibliography/corpus file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Database + date documented.

- Flagged items include recommended action.

- Save all generated artifacts to:
  - benchmark_results/T393_retraction_check/


## Expected Output

Expected output artifact(s):

- `retraction_flags.csv`

- `actions.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
