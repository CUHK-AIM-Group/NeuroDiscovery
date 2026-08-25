# T216_bidskit_conversion: BIDSKIT Conversion Workflow
## Task Description

Convert a dataset to BIDS with BIDSKIT: generate the protocol translation file from a first pass, edit mappings, and run the final conversion.

## Input Requirement

Required input(s):

- DICOM or sourcedata NIfTI/JSON directories (required)


If any required input is missing, return:

- Missing required input


## Constraints

- `Protocol_Translator.json` must be versioned with the output.

- Document any series excluded and why.

- Save all generated artifacts to:
  - benchmark_results/T216_bidskit_conversion/


## Expected Output

Expected output artifact(s):

- BIDS tree

- `Protocol_Translator.json`

- `excluded_series.csv` + reason


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
