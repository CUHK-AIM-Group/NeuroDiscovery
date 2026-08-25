# T364_prisma_fulltext_retrieval: Full-Text Retrieval for Screening
## Task Description

For an included-paper list, attempt full-text PDF retrieval via Unpaywall/open-access endpoints; record OA status and retrieval success per paper.

## Input Requirement

Required input(s):

- Included-paper list from screening (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Unpaywall API with email configured.

- Never bypass paywalls; OA only.

- Save all generated artifacts to:
  - benchmark_results/T364_prisma_fulltext_retrieval/


## Expected Output

Expected output artifact(s):

- `fulltext_status.csv`

- `retrieved/` PDF list


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
