# T129_dvc_data_versioning: DVC Data Versioning for Derivatives
## Task Description

Initialize DVC in an existing analysis repository, place a derivatives
directory under version control with a configured remote, and demonstrate a
checkout/repro round-trip.

## Input Requirement

Required input(s):

- Path to the repository containing the derivatives directory (required)

If any required input is missing, return:

- Missing required input

## Constraints

- `dvc init`, `dvc add derivatives/`, configure a remote (local-path remote
  is acceptable for the benchmark), `dvc push`.
- Git-track the `.dvc` files and `.gitignore` changes; data itself must NOT
  be committed to git.
- Demonstrate `dvc pull` restoring the directory after a clean checkout.
- Save all generated artifacts to:
  - benchmark_results/T129_dvc_data_versioning/

## Expected Output

Expected output artifact(s):

- `derivatives.dvc` and updated `.gitignore`
- `dvc_remote_config.txt` + push/pull logs
- `roundtrip_report.json` (file count + hash before vs. after pull)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
