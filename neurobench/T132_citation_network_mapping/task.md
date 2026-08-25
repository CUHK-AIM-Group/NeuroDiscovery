# T132_citation_network_mapping: Citation Network Mapping from a Seed Paper
## Task Description

Starting from a seed paper, build its one-hop citation network (references +
citing papers) using open scholarly APIs, and produce a structured graph plus
a short landscape summary.

## Input Requirement

Required input(s):

- Seed paper identifier (DOI or Semantic Scholar / OpenAlex ID, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use open APIs only (OpenAlex, Semantic Scholar, Crossref); respect rate
  limits.
- Cap the network at 200 nodes; selection criterion must be documented
  (e.g. top-cited citing works).
- Save all generated artifacts to:
  - benchmark_results/T132_citation_network_mapping/

## Expected Output

Expected output artifact(s):

- `citation_graph.json` (`nodes`: id/title/year/venue/citation_count,
  `edges`: source/target/type)
- `landscape_summary.md` (top venues, publication-year histogram in text
  form, 5-sentence narrative of the research landscape)
- Optional `citation_graph.png` visualization

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
