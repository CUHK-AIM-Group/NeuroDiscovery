# T145_fslmaths_roi_mask: FSL fslmaths ROI Mask Construction
## Task Description

From a probabilistic atlas map, construct a binary ROI mask at a given probability threshold using FSL `fslmaths`, and report mask size and overlap with a subject-specific mask if provided.

## Input Requirement

Required input(s):

- Probabilistic atlas NIfTI (4D or single map, required)

- Probability threshold (default 0.25, configurable)

- Subject mask (optional)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `fslmaths -thr` + `-bin`; document exact command line.

- Threshold must be stated in the output filenames.

- Save all generated artifacts to:
  - benchmark_results/T145_fslmaths_roi_mask/


## Expected Output

Expected output artifact(s):

- `roi_thr{p}.nii.gz` (binary mask)

- `mask_stats.json` (voxel count, volume ml, overlap metrics)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
