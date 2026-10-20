# T237_git_annex_s3_sync: git-annex S3 Sync
## Task Description

Configure a git-annex special remote on S3 (or local-path stand-in) and sync large files, verifying availability counts.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Config/metadata inputs as stated (required)


If any required input is missing, return:

- Missing required input


## Constraints

- `git annex whereis` report before/after sync.

- Credentials via environment only; none in repo files.

- Save all generated artifacts to:
  - benchmark_results/T237_git_annex_s3_sync/


## Expected Output

Expected output artifact(s):

- `annex_sync_report.txt`

- `whereis_summary.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
