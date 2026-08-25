# T178_feat_group_flame: FSL FEAT Group-Level FLAME
## Task Description

Run a group-level mixed-effects analysis (FLAME 1+2) in FSL FEAT over first-level COPE directories, with a two-group or one-sample design.

## Input Requirement

Required input(s):

- First-level FEAT directories for all subjects (required)

- Group design matrix + contrasts (required)


If any required input is missing, return:

- Missing required input


## Constraints

- FLAME 1+2; cluster threshold z>3.1, corrected p<0.05 unless the design dictates otherwise.

- Document the exact `design.fsf` settings.

- Save all generated artifacts to:
  - benchmark_results/T178_feat_group_flame/


## Expected Output

Expected output artifact(s):

- Group FEAT directory with thresholded zstat maps

- `rendered_zstat.png` (MNI slices)

- `design.png` + `design.mat`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
