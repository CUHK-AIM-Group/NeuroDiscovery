# T139_afni_skullstrip: AFNI 3dSkullStrip Brain Extraction
## Task Description

Extract the brain from a T1w volume using AFNI `3dSkullStrip`, then visually and quantitatively compare the mask against a reference mask if one is provided.

## Input Requirement

Required input(s):

- Subject T1w NIfTI file (required)

- Reference brain mask (optional, for Dice comparison)


If any required input is missing, return:

- Missing required input


## Constraints

- Use AFNI `3dSkullStrip` (document the AFNI version).

- Report mask volume in ml.

- Save all generated artifacts to:
  - benchmark_results/T139_afni_skullstrip/


## Expected Output

Expected output artifact(s):

- `brain.nii.gz` (skull-stripped T1w)

- `brain_mask.nii.gz`

- `qc_overlay.png` (mask edges over T1w, 3 orthogonal slices)

- Dice score vs. reference mask if provided (in metadata JSON)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
