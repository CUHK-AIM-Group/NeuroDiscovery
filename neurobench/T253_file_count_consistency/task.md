# T253_file_count_consistency: File Count Consistency Check
## Task Description

Check internal consistency: per-subject file counts by modality against the dataset median; flag outliers (missing runs, doubled runs).

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Outlier rule: |count - median| > 0 flagged with detail.

- No modifications; report only.

- Save all generated artifacts to:
  - benchmark_results/T253_file_count_consistency/


## Expected Output

Expected output artifact(s):

- `file_count_report.csv`

- `outlier_subjects.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
