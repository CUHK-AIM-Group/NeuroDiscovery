# T227_dicom_sort_by_series: DICOM Sort by Series
## Task Description

Sort a flat DICOM dump into Patient/Study/Series directories using header fields, and emit a series-inventory table.

## Input Requirement

Required input(s):

- DICOM directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Sorting key: StudyInstanceUID/SeriesInstanceUID; keep original files untouched (copy or link).

- Save all generated artifacts to:
  - benchmark_results/T227_dicom_sort_by_series/


## Expected Output

Expected output artifact(s):

- Sorted directory tree

- `series_inventory.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
