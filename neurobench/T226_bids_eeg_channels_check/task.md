# T226_bids_eeg_channels_check: EEG-BIDS channels.tsv Check
## Task Description

Validate EEG-BIDS `channels.tsv` against the recording: channel count match, reference electrode declared, and status column present.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use the EEG-BIDS spec for required columns.

- Cross-check with the actual FIF/EDF channel list.

- Save all generated artifacts to:
  - benchmark_results/T226_bids_eeg_channels_check/


## Expected Output

Expected output artifact(s):

- `channels_check.csv` per subject

- `channels_fixes.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
