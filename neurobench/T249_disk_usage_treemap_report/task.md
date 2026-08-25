# T249_disk_usage_treemap_report: Dataset Disk Usage Report
## Task Description

Profile disk usage of a dataset: per-directory sizes, top-20 largest files, format breakdown (DICOM/NIfTI/derivatives), and a text or HTML treemap.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Remote/credentials where applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Read-only; never modify the dataset.

- Identify redundant candidates (e.g. uncompressed duplicates) as suggestions only.

- Save all generated artifacts to:
  - benchmark_results/T249_disk_usage_treemap_report/


## Expected Output

Expected output artifact(s):

- `disk_usage_report.html`

- `largest_files.csv`

- `cleanup_suggestions.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
