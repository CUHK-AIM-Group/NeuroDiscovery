# Benchmark Test Case 51: Nilearn Seed-to-Voxel Correlation

## Task Description

Load local preprocessed BOLD and seed coordinates, compute seed-to-voxel correlation z-map.

## Input Requirement

Required input(s):

- Preprocessed BOLD image (required)
- Seed coordinate(s) (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use Nilearn-compatible workflow and functions.
- Save all generated artifacts to:
  - benchmark_results/T74_nilearn_seed_corr/

## Expected Output

Expected output artifact(s):

- seed_zmap.nii.gz

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
