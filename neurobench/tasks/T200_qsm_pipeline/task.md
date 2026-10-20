# T200_qsm_pipeline: QSM Reconstruction Pipeline
## Task Description

Run quantitative susceptibility mapping from multi-echo GRE: phase unwrapping, background-field removal, dipole inversion, and QSM map in MNI space.

## Input Requirement

Required input(s):

- Multi-echo GRE magnitude + phase NIfTIs (required)

- T1w for registration (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use SEPIA or equivalent open pipeline; document each algorithm choice (unwrap, BFR, inversion).

- Report ROI susceptibility means for deep GM structures.

- Save all generated artifacts to:
  - benchmark_results/T200_qsm_pipeline/


## Expected Output

Expected output artifact(s):

- `qsm_mni.nii.gz`

- `roi_susceptibility.csv`

- `qsm_qc.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
