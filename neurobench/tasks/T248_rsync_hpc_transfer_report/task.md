# T248_rsync_hpc_transfer_report: HPC rsync Transfer + Report
## Task Description

Transfer a dataset to/from an HPC cluster with rsync: resumable, partial-dir friendly, with a post-transfer verification pass.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Remote/credentials where applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `rsync -a --partial --info=progress2` or equivalent.

- Verification: file count + byte size + spot checksums (>= 1% of files).

- Save all generated artifacts to:
  - benchmark_results/T248_rsync_hpc_transfer_report/


## Expected Output

Expected output artifact(s):

- `transfer_log.txt`

- `verification_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
