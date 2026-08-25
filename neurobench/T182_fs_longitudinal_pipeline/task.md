# T182_fs_longitudinal_pipeline: FreeSurfer Longitudinal Pipeline
## Task Description

Run the FreeSurfer longitudinal stream for one subject with 2+ time points: cross-sectional recon-all per TP, unbiased base template, then longitudinal runs, and extract longitudinal volume/thickness changes.

## Input Requirement

Required input(s):

- T1w NIfTIs for >= 2 time points (required)

- FreeSurfer license (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `recon-all -base` + `-long` stream; document FreeSurfer version.

- Do NOT average time points cross-sectionally.

- Save all generated artifacts to:
  - benchmark_results/T182_fs_longitudinal_pipeline/


## Expected Output

Expected output artifact(s):

- Base + longitudinal recon directories

- `longitudinal_change.csv` (per-region annualized change)

- `qc_snapshots.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
