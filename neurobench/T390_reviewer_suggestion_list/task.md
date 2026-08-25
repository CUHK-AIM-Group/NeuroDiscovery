# T390_reviewer_suggestion_list: Reviewer Suggestion List
## Task Description

Suggest qualified reviewers for the manuscript: candidate pool from cited authors + field experts, screened for conflicts (co-authorship, same institution).

## Input Requirement

Required input(s):

- Project abstract/manuscript draft (required)

- Corpus files as applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Conflict screen documented per candidate.

- 8-12 candidates with rationale and contact field placeholders.

- Save all generated artifacts to:
  - benchmark_results/T390_reviewer_suggestion_list/


## Expected Output

Expected output artifact(s):

- `reviewer_suggestions.md`

- `conflict_screen.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
