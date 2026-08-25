# T252_modality_coverage_matrix: Modality Coverage Matrix
## Task Description

Build a subject x modality coverage matrix (T1w/T2w/BOLD/DWI/ASL/PET/ EEG/MEG) for a dataset, with counts and a heatmap render.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Modalities detected from BIDS suffixes, not folder names.

- Heatmap exported as PNG.

- Save all generated artifacts to:
  - benchmark_results/T252_modality_coverage_matrix/


## Expected Output

Expected output artifact(s):

- `modality_matrix.csv`

- `modality_heatmap.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
