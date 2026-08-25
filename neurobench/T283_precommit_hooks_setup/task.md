# T283_precommit_hooks_setup: Pre-Commit Hooks Setup
## Task Description

Configure pre-commit with black/ruff (or flake8), trailing-whitespace and large-file checks, and run it across the repo.

## Input Requirement


- No interactive input.


## Constraints

- Config committed as `.pre-commit-config.yaml`.

- One full-repo pass; fixes committed separately from config.

- Save all generated artifacts to:
  - benchmark_results/T283_precommit_hooks_setup/


## Expected Output

Expected output artifact(s):

- `.pre-commit-config.yaml`

- `first_run_report.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
