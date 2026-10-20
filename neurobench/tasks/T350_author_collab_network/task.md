# T350_author_collab_network: Author Collaboration Network
## Task Description

Build the co-authorship network around a seed author or paper set: nodes = authors, edges = co-authored papers, with community detection.

## Input Requirement

Required input(s):

- Seed paper list or query (required)

- Screening criteria where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Cap at 500 authors; selection rule documented.

- Disambiguate authors by OpenAlex/ORCID where possible.

- Save all generated artifacts to:
  - benchmark_results/T350_author_collab_network/


## Expected Output

Expected output artifact(s):

- `collab_network.json`

- `top_collaborators.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
