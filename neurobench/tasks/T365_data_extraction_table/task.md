# T365_data_extraction_table: Data Extraction Table Builder
## Task Description

From full-texts or abstracts of included papers, build the data extraction table for a meta-analysis: sample size, age, sex ratio, scanner, preprocessing, main finding.

## Input Requirement

Required input(s):

- Included-paper list from screening (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Fields fixed in a schema file.

- Uncertain values marked, not invented.

- Save all generated artifacts to:
  - benchmark_results/T365_data_extraction_table/


## Expected Output

Expected output artifact(s):

- `extraction_table.csv`

- `extraction_notes.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
