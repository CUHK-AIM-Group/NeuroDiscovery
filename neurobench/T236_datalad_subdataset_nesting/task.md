# T236_datalad_subdataset_nesting: DataLad Subdataset Nesting
## Task Description

Restructure a monolithic dataset into nested DataLad subdatasets (raw / derivatives / per-site), registering each in the parent.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Config/metadata inputs as stated (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Each subdataset gets its own `.gitmodules` entry and README.

- History preservation not required; document the cut.

- Save all generated artifacts to:
  - benchmark_results/T236_datalad_subdataset_nesting/


## Expected Output

Expected output artifact(s):

- Nested dataset structure

- `subdataset_map.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
