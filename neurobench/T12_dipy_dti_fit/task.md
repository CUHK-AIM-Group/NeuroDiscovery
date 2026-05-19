# Benchmark Test Case 12: DTI Fit on Selected b-values

## Task Description

Fit diffusion tensor on selected b-value volumes.

Expected workflow:

1. Load DWI, bvals, bvecs, and brain mask
2. Select target b-value subset for fitting
3. Perform tensor fitting in-mask
4. Save fitting outputs for downstream metric calculation

## Input Requirement

Required inputs:

- DWI 4D NIfTI
- `.bval`
- `.bvec`
- `brain_mask.nii.gz` (from previous step)

If required inputs are missing, return:

- `Missing required input`

## Constraints

- Must explicitly describe selected b-value strategy.
- Must fit tensor model on selected volumes.
- Save output artifact(s) to:
  - `benchmark_results/T12_dti_fit/`

## Expected Output

Recommended outputs:

- Tensor fit artifact(s) for downstream metrics
- `result_YYYYMMDD_HHMMSS.json`

## Evaluation

- This test case is **manually evaluated**.
