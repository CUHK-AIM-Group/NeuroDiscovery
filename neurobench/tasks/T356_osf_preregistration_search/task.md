# T356_osf_preregistration_search: OSF Preregistration Search
## Task Description

Search OSF for preregistered studies on resting-state fMRI and depression; extract hypothesis, sample size plan, and analysis plan summaries.

## Input Requirement


- No interactive input.


## Constraints

- OSF API or scrape with documented rate limits.

- Summaries structured per preregistration.

- Save all generated artifacts to:
  - benchmark_results/T356_osf_preregistration_search/


## Expected Output

Expected output artifact(s):

- `preregistrations.csv`

- `prereg_summary.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
