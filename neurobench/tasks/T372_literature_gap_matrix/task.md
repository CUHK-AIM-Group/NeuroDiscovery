# T372_literature_gap_matrix: Literature Gap Matrix
## Task Description

Build a topic x method matrix over a corpus (e.g. disorders x analysis methods) and highlight under-studied cells as candidate gaps.

## Input Requirement

Required input(s):

- Corpus file (paper list or query) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Cell thresholds for 'studied' documented.

- Gap claims must cite cell counts.

- Save all generated artifacts to:
  - benchmark_results/T372_literature_gap_matrix/


## Expected Output

Expected output artifact(s):

- `gap_matrix.csv`

- `gap_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
