# Benchmark Test Case 18: FreeSurfer FLAIR-assisted Segmentation

## Task Description

Load local T1w + FLAIR and run FreeSurfer workflow for improved segmentation quality.

Goal:

- improve aseg/aparc outputs with FLAIR assistance

## Input Requirement

Required:

- local T1w image
- local FLAIR image

If required input is missing, return:

- `Missing required input`

## Expected Workflow

1. Load T1w and FLAIR
2. Run FreeSurfer with FLAIR integration option
3. Export refined aseg/aparc and logs

## Output Requirement

Save artifacts to:

- `benchmark_results/T18_freesurfer_flair_seg/`

Expected key outputs:

- improved `aseg` segmentation
- improved `aparc`-related outputs
- statistics files

## Runtime Note

Long runtime; background execution is acceptable.

## Evaluation

- This test case is **manually evaluated**.
