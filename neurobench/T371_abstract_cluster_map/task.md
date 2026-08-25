# T371_abstract_cluster_map: Abstract Embedding Cluster Map
## Task Description

Cluster a corpus of abstracts into thematic clusters (embedding + HDBSCAN or TF-IDF + k-means), label clusters, and produce a 2D map.

## Input Requirement

Required input(s):

- Corpus file (paper list or query) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Method + parameters documented.

- Cluster labels from top terms.

- Save all generated artifacts to:
  - benchmark_results/T371_abstract_cluster_map/


## Expected Output

Expected output artifact(s):

- `clusters.csv`

- `cluster_map.png`

- `cluster_labels.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
