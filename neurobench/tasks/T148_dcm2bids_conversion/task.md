# T148_dcm2bids_conversion: dcm2bids Single-Session Conversion
## Task Description

Convert one session of DICOMs to BIDS using `dcm2bids` with a written configuration file, then validate the output.

## Input Requirement

Required input(s):

- DICOM directory for one session (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Write and keep the `dcm2bids` config JSON in the output directory.

- Run `dcm2bids_helper` first and record the helper output used to build the config.

- Output must pass `bids-validator` with no errors.

- Save all generated artifacts to:
  - benchmark_results/T148_dcm2bids_conversion/


## Expected Output

Expected output artifact(s):

- BIDS session tree (`sub-*/ses-*`)

- `dcm2bids_config.json`

- `bids_validation_report.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
