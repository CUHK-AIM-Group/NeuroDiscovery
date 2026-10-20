# T173_petprep_pipeline: PETPrep End-to-End Pipeline
## Task Description

Run PETPrep on a BIDS dataset with PET data: motion correction, coregistration to T1w, and uptake-ratio images with the reference region documented.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)

- FreeSurfer license file (required)

- Reference-region specification (required)


If any required input is missing, return:

- Missing required input


## Constraints

- PETPrep container with pinned version.

- State the reference region (e.g. cerebellum) and SUVr formula.

- Save all generated artifacts to:
  - benchmark_results/T173_petprep_pipeline/


## Expected Output

Expected output artifact(s):

- PETPrep derivatives (coregistered PET, SUVr maps)

- `petprep_report.html`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
