# Benchmark Test Case 10: DWI MP-PCA Denoising with DIPY

## Task Description

Apply DIPY-based MP-PCA denoising to a DWI dataset before downstream brain masking/tensor fitting.

Expected workflow:

1. Load DWI 4D NIfTI and corresponding `.bval` / `.bvec`
2. Estimate local noise using DIPY MP-PCA strategy
3. Denoise DWI while preserving original spatial shape and affine
4. Save denoised DWI and a noise-level summary for QC

## Input Requirement

Required inputs:

- DWI 4D NIfTI
- `.bval`
- `.bvec`

If required inputs are missing, return:

- `Missing required input`

## Constraints

- Must use a DIPY denoising method (MP-PCA family).
- Must not change voxel spacing, affine, or volume count.
- Must keep output in the same orientation/space as input.
- Save outputs to:
  - `benchmark_results/T10_dipy_denoise/`

## Expected Output

Required outputs:

- `dwi_denoised.nii.gz`

Recommended outputs:

- `noise_sigma_map.nii.gz` (or equivalent noise-estimation output)
- `result_YYYYMMDD_HHMMSS.json`

Recommended JSON structure:

```json
{
  "metadata": {
    "task": "T10_dipy_denoise",
    "timestamp": "ISO-8601",
    "status": "success or fail"
  },
  "input": {
    "dwi": "path/to/dwi.nii.gz",
    "bval": "path/to/dwi.bval",
    "bvec": "path/to/dwi.bvec"
  },
  "output": {
    "dwi_denoised": "path/to/dwi_denoised.nii.gz",
    "noise_sigma_map": "optional path"
  },
  "checks": {
    "shape_preserved": true,
    "affine_preserved": true,
    "volume_count_preserved": true
  }
}
```

## Evaluation

- This test case is **manually evaluated**.
- Manual reviewer checks whether:
  - DIPY denoising was clearly used,
  - denoised DWI is produced,
  - spatial metadata and volume count are preserved,
  - output paths and QC summary are clear.
