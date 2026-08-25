# T387_survey_comparison_table: Survey Comparison Table Auto-Build
## Task Description

Auto-build the survey comparison table: papers x fields (year, data, method, sample size, metric, result), with missing fields flagged.

## Input Requirement

Required input(s):

- Project abstract/manuscript draft (required)

- Corpus files as applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Field schema provided or derived from 5 exemplars.

- Flag, never invent, missing values.

- Save all generated artifacts to:
  - benchmark_results/T387_survey_comparison_table/


## Expected Output

Expected output artifact(s):

- `comparison_table.csv`

- `table_notes.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
