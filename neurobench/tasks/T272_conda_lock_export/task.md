# T272_conda_lock_export: Conda-Lock Reproducible Lockfile
## Task Description

Generate a fully reproducible conda-lock lockfile from an existing environment YAML and demonstrate a clean-room reinstall.

## Input Requirement


- No interactive input.


## Constraints

- Use conda-lock; lockfile per platform if applicable.

- Clean-room test: new env from lockfile only.

- Save all generated artifacts to:
  - benchmark_results/T272_conda_lock_export/


## Expected Output

Expected output artifact(s):

- `conda-lock.yml`

- `reinstall_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
