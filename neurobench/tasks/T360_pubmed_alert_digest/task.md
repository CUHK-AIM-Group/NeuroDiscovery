# T360_pubmed_alert_digest: PubMed Alert Digest Builder
## Task Description

Emulate a PubMed alert: run a saved query for the last 7 days, diff against the previous week's results, and emit only new hits.

## Input Requirement


- No interactive input.


## Constraints

- State persisted between runs (JSON).

- Diff logic by PMID.

- Save all generated artifacts to:
  - benchmark_results/T360_pubmed_alert_digest/


## Expected Output

Expected output artifact(s):

- `pubmed_digest.md`

- `seen_pmids.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
