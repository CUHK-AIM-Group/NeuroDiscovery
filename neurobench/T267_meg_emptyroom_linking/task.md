# T267_meg_emptyroom_linking: MEG Empty-Room Linking
## Task Description

Link each MEG session to its empty-room noise recording by date proximity and BIDS `AssociatedEmptyRoom` fields; repair missing links.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Proximity window configurable (default same day).

- Repairs limited to JSON sidecar fields.

- Save all generated artifacts to:
  - benchmark_results/T267_meg_emptyroom_linking/


## Expected Output

Expected output artifact(s):

- `emptyroom_links.csv`

- `meg_linking_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
