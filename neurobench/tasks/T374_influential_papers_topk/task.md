# T374_influential_papers_topk: Most Influential Papers (Field-Normalized)
## Task Description

Rank corpus papers by field-normalized influence (citations per year or OpenAlex percentile) and produce an annotated top-20 list.

## Input Requirement

Required input(s):

- Corpus file (paper list or query) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Normalization method documented.

- One-line significance note per paper.

- Save all generated artifacts to:
  - benchmark_results/T374_influential_papers_topk/


## Expected Output

Expected output artifact(s):

- `top20_papers.md`

- `influence_scores.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
