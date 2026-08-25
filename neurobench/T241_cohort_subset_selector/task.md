# T241_cohort_subset_selector: Cohort Subset Selector
## Task Description

Apply inclusion/exclusion criteria (age range, sex, diagnosis, site, QC pass) to a participants table and emit the selected subject list with per-criterion attrition counts.

## Input Requirement

Required input(s):

- Participants/phenotypic table(s) (required)

- Criteria or config file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Criteria given as a small YAML file kept with output.

- Attrition reported in PRISMA style (n removed per criterion).

- Save all generated artifacts to:
  - benchmark_results/T241_cohort_subset_selector/


## Expected Output

Expected output artifact(s):

- `selected_subjects.txt`

- `attrition_report.md`

- `cohort_summary.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
