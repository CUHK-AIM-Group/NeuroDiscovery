# T222_bids_derivatives_atlas_check: BIDS-Derivatives Atlas Compliance Check
## Task Description

Check a derivatives directory against BIDS-Derivatives conventions: required `dataset_description.json` fields (`GeneratedBy`), spatial reference metadata, and naming rules.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use the BIDS-Derivatives spec; cite the rule for each finding.

- Validator-plus-custom-checks approach documented.

- Save all generated artifacts to:
  - benchmark_results/T222_bids_derivatives_atlas_check/


## Expected Output

Expected output artifact(s):

- `derivatives_compliance.csv`

- `fix_suggestions.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
