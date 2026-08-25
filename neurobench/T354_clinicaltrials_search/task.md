# T354_clinicaltrials_search: ClinicalTrials.gov Search: TMS Depression
## Task Description

Search ClinicalTrials.gov for completed/ongoing trials on TMS for depression with connectivity-based targeting; extract structured trial data.

## Input Requirement


- No interactive input.


## Constraints

- Use the ClinicalTrials.gov API v2.

- Fields: NCT, status, phase, enrollment, intervention, primary outcome, results availability.

- Save all generated artifacts to:
  - benchmark_results/T354_clinicaltrials_search/


## Expected Output

Expected output artifact(s):

- `trials.csv`

- `trial_landscape.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
