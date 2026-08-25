# T185_spm_dartel_template: SPM DARTEL Study Template
## Task Description

Build a study-specific DARTEL template from segmented GM/WM images, normalize subjects to MNI via the template, and report template sharpness across iterations.

## Input Requirement

Required input(s):

- SPM segmentation outputs (c1/c2) for >= 10 subjects (required)


If any required input is missing, return:

- Missing required input


## Constraints

- `spm_dartel_template` with default 6 outer iterations.

- Visualize template evolution across iterations.

- Save all generated artifacts to:
  - benchmark_results/T185_spm_dartel_template/


## Expected Output

Expected output artifact(s):

- `Template_0..6.nii` series

- Flow fields + MNI-normalized tissue maps

- `template_evolution.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
