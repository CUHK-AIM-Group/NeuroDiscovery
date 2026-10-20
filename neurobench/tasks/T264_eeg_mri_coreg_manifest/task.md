# T264_eeg_mri_coreg_manifest: EEG-MRI Coregistration Manifest
## Task Description

Build a manifest pairing each EEG recording with the subject's T1w for coregistration, including fiducial availability status.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Fiducial status from the EEG sidecar files.

- Missing-fiducial recordings listed separately.

- Save all generated artifacts to:
  - benchmark_results/T264_eeg_mri_coreg_manifest/


## Expected Output

Expected output artifact(s):

- `eeg_mri_manifest.csv`

- `fiducial_status.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
