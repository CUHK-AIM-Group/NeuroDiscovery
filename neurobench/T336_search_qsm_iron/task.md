# T336_search_qsm_iron: Academic Search: Quantitative Susceptibility Mapping Iron Neurodegeneration
## Task Description

Search for the most recent papers related to **"quantitative susceptibility mapping iron neurodegeneration"** from multiple academic platforms (arXiv, PubMed, Semantic Scholar), deduplicate across platforms, and save structured results.

## Input Requirement


- No interactive input.


## Constraints

- Time range: last 180 days.

- 20 papers per platform (60 total minimum), newest first.

- Deduplicate by DOI/title across platforms.

- Tolerate partial platform failures without failing the run.

- Save all generated artifacts to:
  - benchmark_results/T336_search_qsm_iron/


## Expected Output

Expected output artifact(s):

- `results.json` (metadata + per-platform paper lists with title/authors/published/url/abstract/doi)

- `search_summary.md` (query, counts, date coverage)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
