# T344_syst_gnn_alzheimer: Systematic Search Protocol: Graph Neural Network Alzheimer Classification Mri
## Task Description

Run a systematic-style multi-database search for "graph neural network Alzheimer classification MRI": document the query strings per database (PubMed, Web of Science alternatives like OpenAlex, Scopus-alternative Semantic Scholar), apply date/language filters, deduplicate, and emit the counts trail.

## Input Requirement


- No interactive input.


## Constraints

- Query strings documented verbatim per database.

- Last 5 years, English only; state filters explicitly.

- Counts trail: retrieved per DB -> deduplicated -> final.

- Save all generated artifacts to:
  - benchmark_results/T344_syst_gnn_alzheimer/


## Expected Output

Expected output artifact(s):

- `search_protocol.md` (queries + filters)

- `merged_results.csv`

- `counts_trail.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
