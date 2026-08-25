# T261_citation_cff_generator: CITATION.cff Generator
## Task Description

Generate a valid CITATION.cff for the dataset/codebase and verify it with cffconvert.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Contributors/criteria files where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Validate with `cffconvert --validate`.

- Authors/ORCIDs from the provided contributors file only.

- Save all generated artifacts to:
  - benchmark_results/T261_citation_cff_generator/


## Expected Output

Expected output artifact(s):

- `CITATION.cff`

- `cffconvert_validation.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
