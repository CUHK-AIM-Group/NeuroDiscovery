# T245_longitudinal_session_ordering: Longitudinal Session Ordering
## Task Description

Order longitudinal sessions chronologically from acquisition dates (DICOM/JSON metadata), relabel `ses-` labels if needed, and verify consistent intervals.

## Input Requirement

Required input(s):

- Participants/phenotypic table(s) (required)

- Criteria or config file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Relabeling performed only via copy/rename plan; original kept.

- Sessions without dates flagged, not guessed.

- Save all generated artifacts to:
  - benchmark_results/T245_longitudinal_session_ordering/


## Expected Output

Expected output artifact(s):

- `session_order.csv`

- `relabel_plan.json`

- `interval_check.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
