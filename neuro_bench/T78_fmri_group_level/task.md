# T78_fmri_group_level: fMRI Group-Level Analysis

## Objective
Run FSL randomise or FEAT group-level analysis across multiple subjects

## Inputs
First-level cope/varcope files from multiple subjects and writable output location

If the required first-level inputs are missing, explicitly report `Missing required input` instead of changing the task to a different analysis type.

## Outputs
Group-level statistical maps and contrast estimates

## Key Points
- Stack first-level contrasts across subjects
- Run one-sample (or two-sample) t-test
- Apply permutation testing (randomise) for robust inference
- Generate group-level Z-stat maps
- Perform multiple comparison correction (threshold-free cluster enhancement)
- Identify significant clusters and peaks
- Generate group-level contrasts and parameter estimates
- Output statistical maps in NIFTI format and MNI coordinates
- Keep the task on the group-level inference step; do not restart subject-level preprocessing, first-level modeling, or unrelated connectivity workflows

## Evaluation Criteria
- Task completion verified by presence of required output files
- Output files must be in correct format (NIfTI, CSV, TSV, NPZ where applicable)
- Statistical maps must contain valid numerical data with proper dimensions
- No errors during processing, comprehensive logs generated
