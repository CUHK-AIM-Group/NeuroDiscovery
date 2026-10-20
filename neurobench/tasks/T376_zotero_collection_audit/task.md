# T376_zotero_collection_audit: Zotero Collection Audit
## Task Description

Audit a Zotero export (or API collection): duplicate items, missing DOIs, missing PDFs, and incomplete metadata, with a fix list.

## Input Requirement

Required input(s):

- Zotero export / .bib / PDF folder as applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Duplicates by DOI then fuzzy title.

- Fix list grouped by issue type.

- Save all generated artifacts to:
  - benchmark_results/T376_zotero_collection_audit/


## Expected Output

Expected output artifact(s):

- `zotero_audit.csv`

- `fix_list.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
