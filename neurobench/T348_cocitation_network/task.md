# T348_cocitation_network: Co-Citation Network Analysis
## Task Description

Build a co-citation network for a seed corpus: papers frequently cited together form clusters; identify the intellectual base of the topic.

## Input Requirement

Required input(s):

- Seed paper list or query (required)

- Screening criteria where applicable


If any required input is missing, return:

- Missing required input


## Constraints

- Seed corpus from a provided reference list (DOIs).

- Threshold for co-citation edges documented (e.g. >= 3).

- Save all generated artifacts to:
  - benchmark_results/T348_cocitation_network/


## Expected Output

Expected output artifact(s):

- `cocitation_graph.json`

- `cluster_summary.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
