# T258_dataset_description_writer: dataset_description.json Writer
## Task Description

Author or repair `dataset_description.json` files for a dataset and its derivatives: correct BIDSVersion, Name, DatasetType, GeneratedBy blocks.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Contributors/criteria files where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Derivatives must include GeneratedBy with tool + version.

- Validate with bids-validator afterwards.

- Save all generated artifacts to:
  - benchmark_results/T258_dataset_description_writer/


## Expected Output

Expected output artifact(s):

- Updated JSON files

- `description_diff.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
