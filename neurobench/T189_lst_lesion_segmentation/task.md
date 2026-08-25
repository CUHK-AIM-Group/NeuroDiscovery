# T189_lst_lesion_segmentation: LST Lesion Segmentation (SPM)
## Task Description

Run SPM/LST lesion prediction (LPA) on FLAIR images for a small cohort and produce per-subject lesion maps + volumes.

## Input Requirement

Required input(s):

- FLAIR NIfTIs for the cohort (required)

- T1w NIfTIs (optional, used by LGA variant)


If any required input is missing, return:

- Missing required input


## Constraints

- LST toolbox via SPM batch; LPA unless T1w provided (then document LGA).

- Threshold lesions at the recommended probability; state it.

- Save all generated artifacts to:
  - benchmark_results/T189_lst_lesion_segmentation/


## Expected Output

Expected output artifact(s):

- Per-subject `ples_*.nii` maps

- `lesion_volumes.csv`

- `lst_batch_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
