# T174_mne_rest_full_pipeline: MNE Resting-State EEG Full Pipeline
## Task Description

Full resting-state EEG pipeline with MNE-Python: preprocessing (filter, ICA), sensor-level spectral analysis (PSD + band power), and connectivity (wPLI) between regions.

## Input Requirement

Required input(s):

- Resting-state EEG recording (required)

- Channel montage/positions (required)


If any required input is missing, return:

- Missing required input


## Constraints

- MNE-Python throughout; band-pass 1-40 Hz.

- Report PSD per canonical band (delta..gamma) per channel group.

- Connectivity: wPLI on cleaned epochs; state epoch length.

- Save all generated artifacts to:
  - benchmark_results/T174_mne_rest_full_pipeline/


## Expected Output

Expected output artifact(s):

- `psd_by_band.csv`

- `wpli_connectome.npy` + matrix PNG

- `pipeline_report.html` (MNE Report)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
