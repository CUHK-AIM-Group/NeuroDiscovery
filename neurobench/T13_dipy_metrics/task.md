# Benchmark Test Case 13: DTI Core Metrics

## Task Description

Compute and save four core DTI metrics in NIfTI format:

- FA (fractional anisotropy)
- MD (mean diffusivity)
- AD (axial diffusivity)
- RD (radial diffusivity)

## Numerical Safety Requirements

Must include the following protections:

1. Negative eigenvalues are clipped to 0
2. Non-finite values (`NaN`, `Inf`) are set to 0
3. Voxels outside mask are set to 0

## Input Requirement

Required inputs:

- Tensor fitting outputs (from previous step)
- Brain mask NIfTI

If required input is missing, return:

- `Missing required input`

## Output Requirement

Save outputs to:

- `benchmark_results/T13_dwi_metrics/`

Required files (NIfTI):

- `FA.nii` or `FA.nii.gz`
- `MD.nii` or `MD.nii.gz`
- `AD.nii` or `AD.nii.gz`
- `RD.nii` or `RD.nii.gz`

## Success Criteria

- All four metric files exist
- Shapes are consistent
- Values are finite
- Numerical safety protections are applied
