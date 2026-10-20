# Benchmark Test Case 16: FreeSurfer Fast Subcort + Basic Surface

## Task Description

Load local T1w and run a fast FreeSurfer-oriented workflow targeting:

- fast subcortical segmentation
- basic white/gray matter cortical surfaces

## Input Requirement

Required:

- local T1w image

If required input is missing, return:

- `Missing required input`

## Expected Workflow

1. Load T1w
2. Execute a speed-prioritized pipeline (reduced/full-skip stages as appropriate)
3. Generate aseg and key cortical surfaces

## Output Requirement

Save artifacts to:

- `benchmark_results/T16_freesurfer_fast_subcort/`

Expected key outputs:

- `aseg` segmentation
- white matter surfaces
- pial/gray boundary basic surfaces

## Runtime Note

May run in background.

## Evaluation

- This test case is **manually evaluated**.
