# T351_keyword_cooccurrence_map: Keyword Co-Occurrence Map
## Task Description

Extract author keywords from a corpus and build a keyword co-occurrence map with clusters, revealing topic structure.

## Input Requirement

Required input(s):

- Seed paper list or query (required)

- Screening criteria where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Minimum keyword occurrence 5.

- Clusters labeled by their top keywords.

- Save all generated artifacts to:
  - benchmark_results/T351_keyword_cooccurrence_map/


## Expected Output

Expected output artifact(s):

- `keyword_network.json`

- `keyword_clusters.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
