# T235_datalad_dataset_publish: DataLad Dataset Publish
## Task Description

Publish a BIDS dataset as a DataLad dataset to a sibling (local path or S3), verifying clone-ability from the published sibling.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Config/metadata inputs as stated (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Test the round-trip: fresh `datalad clone` + `datalad get` of a subset.

- Save all generated artifacts to:
  - benchmark_results/T235_datalad_dataset_publish/


## Expected Output

Expected output artifact(s):

- Published sibling URL/path record

- `publish_log.txt`

- `clone_verification.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
