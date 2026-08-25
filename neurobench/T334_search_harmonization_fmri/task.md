# T334_search_harmonization_fmri: Academic Search: Multi-Site Harmonization Resting-State Fmri
## Task Description

Search for the most recent papers related to **"multi-site harmonization resting-state fMRI"** from multiple academic platforms (arXiv, PubMed, Semantic Scholar), deduplicate across platforms, and save structured results.

## Input Requirement


- No interactive input.


## Constraints

- Time range: last 180 days.

- 20 papers per platform (60 total minimum), newest first.

- Deduplicate by DOI/title across platforms.

- Tolerate partial platform failures without failing the run.

- Save all generated artifacts to:
  - benchmark_results/T334_search_harmonization_fmri/


## Expected Output

Expected output artifact(s):

- `results.json` (metadata + per-platform paper lists with title/authors/published/url/abstract/doi)

- `search_summary.md` (query, counts, date coverage)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
