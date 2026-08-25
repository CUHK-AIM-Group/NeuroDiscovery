# T377_bibtex_hygiene_fix: BibTeX Hygiene Repair
## Task Description

Clean a .bib file: consistent entry keys, complete fields (via Crossref lookup), no duplicate entries, valid LaTeX escaping.

## Input Requirement

Required input(s):

- Zotero export / .bib / PDF folder as applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Crossref lookups for DOI-less entries documented.

- Before/after stats reported.

- Save all generated artifacts to:
  - benchmark_results/T377_bibtex_hygiene_fix/


## Expected Output

Expected output artifact(s):

- `references_clean.bib`

- `hygiene_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
