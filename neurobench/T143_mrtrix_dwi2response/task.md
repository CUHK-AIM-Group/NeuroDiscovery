# T143_mrtrix_dwi2response: MRtrix dwi2response (Dhollander)
## Task Description

Estimate fibre response functions from a preprocessed DWI dataset using MRtrix3 `dwi2response dhollander`, suitable for multi-tissue CSD.

## Input Requirement

Required input(s):

- Preprocessed DWI (`dwi.mif`) with bval/bvec (required)

- Brain mask (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `dwi2response dhollander`; document MRtrix3 version.

- Report the number of voxels selected for WM/GM/CSF responses.

- Save all generated artifacts to:
  - benchmark_results/T143_mrtrix_dwi2response/


## Expected Output

Expected output artifact(s):

- `response_wm.txt`, `response_gm.txt`, `response_csf.txt`

- `response_voxels.mif` (selected-voxel mask)

- `response_plot.png` (response function profiles)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
