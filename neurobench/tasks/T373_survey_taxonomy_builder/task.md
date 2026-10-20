# T373_survey_taxonomy_builder: Survey Taxonomy Builder
## Task Description

From a review corpus, propose a taxonomy for the survey paper: 3-5 top-level categories with inclusion rules, every corpus paper assigned.

## Input Requirement

Required input(s):

- Corpus file (paper list or query) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Every paper assigned to exactly one leaf.

- Rules written so a second rater could apply them.

- Save all generated artifacts to:
  - benchmark_results/T373_survey_taxonomy_builder/


## Expected Output

Expected output artifact(s):

- `taxonomy.md`

- `paper_assignments.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
