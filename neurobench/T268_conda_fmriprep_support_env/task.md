# T268_conda_fmriprep_support_env: Conda Env: fMRIPrep Support Tools
## Task Description

Create a fresh conda env with the support tooling needed around fMRIPrep runs (bids-validator, pybids, pandas, niworkflows client libs) and smoke-test imports.

## Input Requirement


- No interactive input.


## Constraints

- Pinned versions via explicit spec file.

- Smoke test: python import check + validator version.

- Save all generated artifacts to:
  - benchmark_results/T268_conda_fmriprep_support_env/


## Expected Output

Expected output artifact(s):

- `environment.yml`

- `smoke_test.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
