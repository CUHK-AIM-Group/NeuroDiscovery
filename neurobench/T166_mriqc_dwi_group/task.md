# T166_mriqc_dwi_group: MRIQC Group QC: diffusion (DWI)
## Task Description

Run MRIQC participant + group level over all DWI scans in a BIDS dataset, producing per-scan IQMs and the group report, then flag outlier scans.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- MRIQC container with pinned version.

- Outlier rule: any IQM more than 1.5x IQR from the group distribution; document the IQM(s) used.

- Save all generated artifacts to:
  - benchmark_results/T166_mriqc_dwi_group/


## Expected Output

Expected output artifact(s):

- Per-participant IQM JSON/TSV + group TSV

- `group_{mod}.html` report

- `outliers.csv` (flagged scans + reason)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
