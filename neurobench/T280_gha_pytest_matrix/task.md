# T280_gha_pytest_matrix: GitHub Actions: Pytest Matrix
## Task Description

Add a GitHub Actions workflow running the test suite across Python versions (3.10-3.12) and OS (ubuntu, macOS), with caching.

## Input Requirement


- No interactive input.


## Constraints

- Matrix defined explicitly; pip caching enabled.

- Badge added to README.

- Save all generated artifacts to:
  - benchmark_results/T280_gha_pytest_matrix/


## Expected Output

Expected output artifact(s):

- `.github/workflows/tests.yml`

- `ci_run_summary.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
