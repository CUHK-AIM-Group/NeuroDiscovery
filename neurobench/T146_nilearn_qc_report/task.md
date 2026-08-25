# T146_nilearn_qc_report: Nilearn fMRI QC Report Generation
## Task Description

Generate a one-page HTML quality-control report for a preprocessed resting-state fMRI run using nilearn plotting: carpet plot, mean image, motion parameters, and FD time series.

## Input Requirement

Required input(s):

- Preprocessed resting-state fMRI 4D NIfTI (required)

- Confounds TSV with motion + framewise displacement (required)

- Brain mask (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use nilearn (`nilearn.plotting`, `nilearn.image`) only; no external QC packages.

- Report mean FD and number of FD > 0.5 mm volumes.

- Save all generated artifacts to:
  - benchmark_results/T146_nilearn_qc_report/


## Expected Output

Expected output artifact(s):

- `qc_report.html` (self-contained)

- `qc_summary.json` (mean FD, scrubbed volumes, tSNR)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
