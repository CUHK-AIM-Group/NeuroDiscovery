# T179_feat_firstlevel_blocked: FSL FEAT First-Level Blocked Design
## Task Description

Full first-level FEAT analysis for one subject: pre-stats (motion correction, smoothing), FILM GLM with a blocked design, registration to structural + MNI.

## Input Requirement

Required input(s):

- Raw task fMRI 4D NIfTI (required)

- T1w structural (required)

- 3-column event files (required)


If any required input is missing, return:

- Missing required input


## Constraints

- FEAT GUI-equivalent `fsf` file kept in outputs; FWHM 5 mm.

- Registration: BBR to T1w, 12-dof to MNI.

- Save all generated artifacts to:
  - benchmark_results/T179_feat_firstlevel_blocked/


## Expected Output

Expected output artifact(s):

- `*.feat` directory (zstats, cope/varcope)

- `report.html` from FEAT

- `design.png`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
