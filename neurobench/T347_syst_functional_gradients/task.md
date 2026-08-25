# T347_syst_functional_gradients: Systematic Search Protocol: Connectopic Mapping Functional Gradients Cortex
## Task Description

Run a systematic-style multi-database search for "connectopic mapping functional gradients cortex": document the query strings per database (PubMed, Web of Science alternatives like OpenAlex, Scopus-alternative Semantic Scholar), apply date/language filters, deduplicate, and emit the counts trail.

## Input Requirement


- No interactive input.


## Constraints

- Query strings documented verbatim per database.

- Last 5 years, English only; state filters explicitly.

- Counts trail: retrieved per DB -> deduplicated -> final.

- Save all generated artifacts to:
  - benchmark_results/T347_syst_functional_gradients/


## Expected Output

Expected output artifact(s):

- `search_protocol.md` (queries + filters)

- `merged_results.csv`

- `counts_trail.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
