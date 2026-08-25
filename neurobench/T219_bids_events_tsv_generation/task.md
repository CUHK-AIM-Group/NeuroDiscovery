# T219_bids_events_tsv_generation: BIDS events.tsv Generation
## Task Description

Generate BIDS-compliant `events.tsv` files from raw stimulus/presentation logs for all task runs in a dataset.

## Input Requirement

Required input(s):

- Raw event logs (Presentation/E-Prime/psychopy, required)

- Task fMRI BIDS tree (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Columns onset/duration/trial_type mandatory; document any extra columns in a JSON sidecar.

- Onsets aligned to scan start (state the trigger convention).

- Save all generated artifacts to:
  - benchmark_results/T219_bids_events_tsv_generation/


## Expected Output

Expected output artifact(s):

- `*_events.tsv` per run

- `events_json_sidecars`

- `event_count_report.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
