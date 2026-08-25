# T309_pip_constraints_file: pip Constraints File Generation
## Task Description

Produce a constraints file that pins the transitive dependency tree of the analysis environment, and verify a clean venv install from it.

## Input Requirement


- No interactive input.


## Constraints

- Constraints include hashes if pip-tools available.

- Clean venv verification log.

- Save all generated artifacts to:
  - benchmark_results/T309_pip_constraints_file/


## Expected Output

Expected output artifact(s):

- `constraints.txt`

- `clean_install_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
