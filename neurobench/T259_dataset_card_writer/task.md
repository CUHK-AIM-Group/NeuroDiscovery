# T259_dataset_card_writer: Dataset Card Writer
## Task Description

Write a dataset card (README) for a neuroimaging dataset: cohort description, acquisition parameters, provenance, license, ethics, citation.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Contributors/criteria files where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Follow the dataset-cards convention (sections in fixed order).

- Every acquisition number traced to the data (no invented values).

- Save all generated artifacts to:
  - benchmark_results/T259_dataset_card_writer/


## Expected Output

Expected output artifact(s):

- `README.md` dataset card

- `sources_checked.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
