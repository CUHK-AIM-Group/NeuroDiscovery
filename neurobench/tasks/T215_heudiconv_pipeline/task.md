# T215_heudiconv_pipeline: HeuDiConv DICOM-to-BIDS Conversion
## Task Description

Convert DICOMs to BIDS with HeuDiConv: write the heuristic file, run conversion for one study, and validate.

## Input Requirement

Required input(s):

- DICOM directories (DICOMDIR or tarballs, required)


If any required input is missing, return:

- Missing required input


## Constraints

- Keep the heuristic `.py` in the output directory.

- Reproducibility: second run must be idempotent (no dupes).

- Save all generated artifacts to:
  - benchmark_results/T215_heudiconv_pipeline/


## Expected Output

Expected output artifact(s):

- BIDS tree

- `heuristic.py`

- `bids_validation_report.txt`

- `heudiconv_log.tsv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
