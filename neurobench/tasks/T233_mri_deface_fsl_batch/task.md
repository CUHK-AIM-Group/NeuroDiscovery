# T233_mri_deface_fsl_batch: FSL fsl_deface Batch
## Task Description

Batch-deface T1w/T2w images with FSL `fsl_deface` and compare mask coverage against pydeface output on a sample.

## Input Requirement

Required input(s):

- BIDS dataset directory (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Document FSL version.

- Comparison subset: at least 3 subjects if pydeface output is available.

- Save all generated artifacts to:
  - benchmark_results/T233_mri_deface_fsl_batch/


## Expected Output

Expected output artifact(s):

- Defaced NIfTIs

- `deface_comparison.md`

- `deface_log.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
