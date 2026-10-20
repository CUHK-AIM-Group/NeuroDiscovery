# T142_ants_apply_transforms: ANTs Apply Transforms: Atlas to Subject Space
## Task Description

Warp a volumetric atlas (e.g. Harvard-Oxford or AAL) from MNI space into subject T1w space using an existing ANTs transform chain, with label-preserving interpolation.

## Input Requirement

Required input(s):

- Atlas NIfTI in MNI space (required)

- Subject T1w NIfTI (required)

- Existing warp + affine from a prior registration (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `antsApplyTransforms -d 3` with `GenericLabel` (nearest-neighbour) interpolation for labels.

- Verify ROI count is preserved after warping.

- Save all generated artifacts to:
  - benchmark_results/T142_ants_apply_transforms/


## Expected Output

Expected output artifact(s):

- `atlas_in_subject.nii.gz`

- `roi_count_check.json` (ROI labels before vs. after)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- Warped atlas must contain the same label set as the input atlas.

- This test case is manually evaluated.
