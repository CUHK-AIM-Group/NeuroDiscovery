# T246_exclusion_log_generator: Exclusion Log Generator
## Task Description

Consolidate QC failure lists from multiple pipeline stages into one exclusion log with stage, reason, and final-inclusion flag per subject.

## Input Requirement

Required input(s):

- Participants/phenotypic table(s) (required)

- Criteria or config file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Each stage's criteria quoted in the log header.

- Final flag = AND of all stages; conflicts flagged.

- Save all generated artifacts to:
  - benchmark_results/T246_exclusion_log_generator/


## Expected Output

Expected output artifact(s):

- `exclusion_log.csv`

- `final_cohort.txt`

- `exclusion_summary.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
