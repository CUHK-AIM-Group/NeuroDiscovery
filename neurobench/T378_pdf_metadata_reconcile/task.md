# T378_pdf_metadata_reconcile: PDF Metadata Reconciliation
## Task Description

Reconcile a folder of PDFs against the .bib: match by title, flag unmatched on both sides, and rename PDFs to the citation key scheme.

## Input Requirement

Required input(s):

- Zotero export / .bib / PDF folder as applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Matching threshold documented; low-confidence matches flagged.

- Renames as a plan first, then executed with log.

- Save all generated artifacts to:
  - benchmark_results/T378_pdf_metadata_reconcile/


## Expected Output

Expected output artifact(s):

- `match_report.csv`

- `rename_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
