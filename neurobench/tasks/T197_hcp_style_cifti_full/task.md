# T197_hcp_style_cifti_full: HCP-Style CIFTI Full-Subject Pipeline
## Task Description

Produce full CIFTI outputs for one subject from fMRIPrep + FreeSurfer results: map rfMRI to 91k grayordinates, denoise, and build a dense connectome, following HCP conventions.

## Input Requirement

Required input(s):

- fMRIPrep + FreeSurfer derivatives for the subject (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Use ciftify or equivalent wb_command steps; 32k fsLR surfaces.

- Document every wb_command call in `cifti_steps.sh`.

- Save all generated artifacts to:
  - benchmark_results/T197_hcp_style_cifti_full/


## Expected Output

Expected output artifact(s):

- `rfMRI.dtseries.nii` (91k grayordinates)

- `dense_connectome.dconn.nii` or ROI connectome

- `cifti_steps.sh`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
