# Benchmark Test Case 15: FreeSurfer Full Pipeline (all)

## Task Description

Load local T1w (required), optionally T2w/FLAIR, and run a full FreeSurfer workflow.

Goal:

- complete cortical reconstruction and volumetric segmentation
- output all major surfaces, segmentations, cortical thickness maps, and stats files

## Input Requirement

Required:

- local T1w image

Optional:

- local T2w image
- local FLAIR image

If required input is missing, return:

- `Missing required input`

## Expected Workflow

1. Initialize FreeSurfer environment
2. Run full pipeline (e.g., `recon-all` full stages)
3. If T2w/FLAIR present, use corresponding refinement options
4. Export output directory and summary logs

## Output Requirement

Save artifacts to:

- `benchmark_results/T15_freesurfer_all/`

Expected outputs include FreeSurfer subject tree with:

- surfaces (`surf/`)
- segmentations (`mri/`)
- thickness maps
- stats files (`stats/`)

## Runtime Note

FreeSurfer runtime is long. Background execution is acceptable.

## Evaluation

- This test case is **manually evaluated**.
