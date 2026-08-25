# T232_pydeface_batch: PyDeface Batch Defacing
## Task Description

Batch-deface all T1w images in a BIDS dataset with pydeface, writing defaced copies to a new dataset tree (originals untouched) with QC renders.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Defaced dataset gets its own `dataset_description.json` noting the defacing step.

- QC: brain mask must survive (render 3-slice overlays).

- Save all generated artifacts to:
  - benchmark_results/T232_pydeface_batch/


## Expected Output

Expected output artifact(s):

- Defaced BIDS tree

- `deface_qc/*.png`

- `deface_log.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
