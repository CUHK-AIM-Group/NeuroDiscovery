# T196_multimodal_qc_pipeline: Multi-Modal QC Aggregation Pipeline
## Task Description

Aggregate quality control across modalities for one BIDS dataset: run or collect MRIQC IQMs for T1w/BOLD/DWI, merge with pipeline success logs, and produce a single subject x modality QC matrix.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)

- Existing pipeline logs (optional)


If any required input is missing, return:

- Missing required input


## Constraints

- One row per subject x modality; status in {pass, warn, fail}.

- Rules for warn/fail must be explicit and reproducible.

- Save all generated artifacts to:
  - benchmark_results/T196_multimodal_qc_pipeline/


## Expected Output

Expected output artifact(s):

- `qc_matrix.csv`

- `qc_dashboard.html` (sortable table + distributions)

- `exclusion_recommendations.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
