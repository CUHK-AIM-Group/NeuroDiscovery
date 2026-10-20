# T172_aslprep_pipeline: ASLPrep End-to-End Pipeline
## Task Description

Run ASLPrep on a BIDS dataset with ASL data: full preprocessing including CBF computation, BASIL partial-volume correction, and QC report.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)

- FreeSurfer license file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- ASLPrep container with pinned version; default options unless justified in the log.

- Note whether pCASL or PASL and handle accordingly.

- Save all generated artifacts to:
  - benchmark_results/T172_aslprep_pipeline/


## Expected Output

Expected output artifact(s):

- ASLPrep derivatives (CBF maps, QC HTML)

- `run_summary.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
