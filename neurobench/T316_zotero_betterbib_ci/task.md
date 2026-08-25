# T316_zotero_betterbib_ci: Zotero Better-BibTeX CI Export
## Task Description

Automate a citation .bib export from Zotero (Better-BibTeX) into the paper repo, with a CI check that all \cite keys resolve.

## Input Requirement


- No interactive input.


## Constraints

- Key format documented.

- CI check greps \cite keys vs. bib.

- Save all generated artifacts to:
  - benchmark_results/T316_zotero_betterbib_ci/


## Expected Output

Expected output artifact(s):

- `references.bib` update flow

- `cite_check.sh`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
