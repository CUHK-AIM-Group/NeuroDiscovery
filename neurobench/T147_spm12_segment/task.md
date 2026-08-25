# T147_spm12_segment: SPM12 Unified Segmentation
## Task Description

Run SPM12 unified segmentation on a T1w image (via MATLAB, Octave, or nipype's SPM interface) producing GM/WM/CSF tissue probability maps.

## Input Requirement

Required input(s):

- Subject T1w NIfTI file (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use SPM12 `spm_preproc` (unified segmentation) with default tissue priors; document SPM version and runtime environment.

- Native-space outputs only; no DARTEL normalization.

- Save all generated artifacts to:
  - benchmark_results/T147_spm12_segment/


## Expected Output

Expected output artifact(s):

- `c1*.nii` GM, `c2*.nii` WM, `c3*.nii` CSF maps

- `tissue_volumes.csv` (per-tissue volume in ml)

- `segment_qc.png` (tissue map overlays)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
