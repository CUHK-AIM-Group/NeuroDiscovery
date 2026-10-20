# T141_ants_n4_biasfield: ANTs N4 Bias Field Correction
## Task Description

Correct intensity inhomogeneity in a T1w image using ANTs `N4BiasFieldCorrection` and quantify the improvement in intensity uniformity within a brain mask.

## Input Requirement

Required input(s):

- Subject T1w NIfTI file (required)

- Brain mask (optional)


If any required input is missing, return:

- Missing required input


## Constraints

- Use `N4BiasFieldCorrection -d 3`; document shrink factor and spline distance if changed from defaults.

- Report coefficient of variation (CV) of white-matter intensity before vs. after correction when a mask is available.

- Save all generated artifacts to:
  - benchmark_results/T141_ants_n4_biasfield/


## Expected Output

Expected output artifact(s):

- `t1w_n4.nii.gz` (bias-corrected image)

- `bias_field.nii.gz` (estimated field)

- `n4_report.json` (parameters + CV before/after)


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
