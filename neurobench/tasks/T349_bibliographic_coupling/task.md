# T349_bibliographic_coupling: Bibliographic Coupling Map
## Task Description

Compute bibliographic coupling among recent papers on a topic (shared references) and cluster them into research fronts.

## Input Requirement

Required input(s):

- Seed paper list or query (required)

- Screening criteria where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Corpus from OpenAlex/Semantic Scholar query; query documented.

- Report the top coupled clusters with representative papers.

- Save all generated artifacts to:
  - benchmark_results/T349_bibliographic_coupling/


## Expected Output

Expected output artifact(s):

- `coupling_edges.csv`

- `research_fronts.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
