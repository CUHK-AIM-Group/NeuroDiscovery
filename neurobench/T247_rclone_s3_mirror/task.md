# T247_rclone_s3_mirror: Rclone S3 Mirror
## Task Description

Mirror a dataset directory to S3-compatible storage with rclone: configure the remote, sync with checksum verification, and produce a transfer report.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Remote/credentials where applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- `--checksum` verification; dry-run output saved before the real sync.

- No credentials in command logs (env/config redacted).

- Save all generated artifacts to:
  - benchmark_results/T247_rclone_s3_mirror/


## Expected Output

Expected output artifact(s):

- `rclone_config_redacted.txt`

- `sync_report.txt`

- `dry_run_diff.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
