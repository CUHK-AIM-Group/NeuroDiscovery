# T177_meg_maxfilter_pipeline: MEG MaxFilter + Preprocessing Pipeline
## Task Description

Preprocess raw MEG with MaxFilter (SSS/tSSS + head-position correction), then filter, annotate bad segments, and produce a sensor QC summary.

## Input Requirement

Required input(s):

- Raw MEG FIF recording (required)

- Empty-room recording for noise covariance (required)


If any required input is missing, return:

- Missing required input


## Constraints

- MaxFilter via `mne.preprocessing.maxwell_filter` or Elekta software; document which.

- tSSS parameters and bad-channel detection logged.

- Save all generated artifacts to:
  - benchmark_results/T177_meg_maxfilter_pipeline/


## Expected Output

Expected output artifact(s):

- `sss_raw.fif` (cleaned recording)

- `noise_cov.fif` from empty room

- `bad_channels.json` + PSD QC plot


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
