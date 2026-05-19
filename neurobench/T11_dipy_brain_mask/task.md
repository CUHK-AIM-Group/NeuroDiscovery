# Benchmark Test Case 11: DWI Brain Mask from b0

## Task Description

After loading DWI, automatically generate `brain_mask.nii.gz` from b0 image(s).

Expected workflow:

1. Identify b0 volume(s) from b-values
2. Build b0 reference (single or mean b0)
3. Generate brain mask automatically
4. Save `brain_mask.nii.gz`

## Input Requirement

Required inputs:

- DWI 4D NIfTI
- `.bval`
- `.bvec`

If required inputs are missing, return:

- `Missing required input`

## Constraints

- No manual masking.
- Use automatic method based on b0 image.
- Save output to:
  - `benchmark_results/T11_dwi_brain_mask/`

## Expected Output

Required output:

- `brain_mask.nii.gz`

Recommended metadata file:

- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- This test case is **manually evaluated**.
