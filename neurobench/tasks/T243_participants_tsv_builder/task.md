# T243_participants_tsv_builder: participants.tsv Builder
## Task Description

Merge multiple phenotypic CSV exports into a single BIDS `participants.tsv`: join on subject ID, harmonize column names, and flag unmatched rows on both sides.

## Input Requirement

Required input(s):

- Participants/phenotypic table(s) (required)

- Criteria or config file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Join rules documented; no row silently dropped.

- Column dictionary emitted as participants.json.

- Save all generated artifacts to:
  - benchmark_results/T243_participants_tsv_builder/


## Expected Output

Expected output artifact(s):

- `participants.tsv`

- `participants.json`

- `merge_conflicts.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
