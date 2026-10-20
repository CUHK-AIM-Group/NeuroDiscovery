# T244_session_matching_report: Cross-Modal Session Matching Report
## Task Description

For each subject, check that required modality sessions (T1w, rest, DWI) exist within the same session label, and report unmatched modality-session pairs.

## Input Requirement

Required input(s):

- Participants/phenotypic table(s) (required)

- Criteria or config file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Matching rules in a config (which modalities required).

- Output distinguishes missing session vs. missing modality.

- Save all generated artifacts to:
  - benchmark_results/T244_session_matching_report/


## Expected Output

Expected output artifact(s):

- `session_matching.csv`

- `unmatched_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
