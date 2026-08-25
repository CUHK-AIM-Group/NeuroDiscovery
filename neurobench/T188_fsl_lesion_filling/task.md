# T188_fsl_lesion_filling: FSL Lesion Filling Pipeline
## Task Description

Fill white-matter lesions in a T1w image using FSL `lesion_filling` prior to downstream volumetric analysis, given a lesion mask from FLAIR segmentation.

## Input Requirement

Required input(s):

- T1w NIfTI (required)

- Lesion mask (e.g. from T71 WMH segmentation) (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use FSL `lesion_filling`; document FSL version.

- QC: filled-region intensity should match surrounding WM (report mean intensities).

- Save all generated artifacts to:
  - benchmark_results/T188_fsl_lesion_filling/


## Expected Output

Expected output artifact(s):

- `t1w_lesion_filled.nii.gz`

- `fill_qc.json` (intensity stats)

- `fill_overlay.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
