# T190_dwi_full_connectome_graph: DWI Tractography-Connectome-Graph Pipeline
## Task Description

Full MRtrix chain from preprocessed DWI: response estimation, multi-shell CSD, ACT tractography (10M), SIFT2, connectome construction with a chosen atlas, and graph-metric extraction.

## Input Requirement

Required input(s):

- Preprocessed DWI + mask + 5TT (required)

- Parcellation atlas in subject space (required)


If any required input is missing, return:

- Missing required input


## Constraints

- MRtrix3 throughout; document every command in `pipeline_log.sh`.

- Connectome weighted by SIFT2 streamline counts (invnode volume normalisation documented).

- Save all generated artifacts to:
  - benchmark_results/T190_dwi_full_connectome_graph/


## Expected Output

Expected output artifact(s):

- `connectome_sift2.csv`

- `graph_metrics.json` (degree, strength, betweenness, efficiency)

- `tractogram_qc.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
