# T242_stratified_split_manifest: Stratified Split Manifest
## Task Description

Create train/validation/test split manifests stratified by site, sex, and age-bin; deterministic with a fixed seed.

## Input Requirement

Required input(s):

- Participants/phenotypic table(s) (required)

- Criteria or config file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Stratification report shows group balance per split.

- Seed logged; re-run with same seed must reproduce identical splits.

- Save all generated artifacts to:
  - benchmark_results/T242_stratified_split_manifest/


## Expected Output

Expected output artifact(s):

- `split_train.txt`, `split_val.txt`, `split_test.txt`

- `stratification_report.csv`

- `split_config.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
