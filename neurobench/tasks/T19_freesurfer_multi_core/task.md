# Benchmark Test Case 19: FreeSurfer Multi-core Acceleration

## Task Description

Load local T1w (same as base FreeSurfer task), run FreeSurfer with multi-core acceleration.

Goal:

- produce outputs equivalent to baseline task
- reduce wall-clock runtime through multi-core settings

## Input Requirement

Required:

- local T1w image

If required input is missing, return:

- `Missing required input`

## Expected Workflow

1. Load T1w
2. Configure multi-core/parallel execution parameters
3. Run pipeline
4. Save outputs and runtime logs

## Output Requirement

Save artifacts to:

- `benchmark_results/T19_freesurfer_multi_core/`

Expected outputs:

- same family as baseline FreeSurfer outputs (surf/mri/stats)
- runtime evidence for multi-core configuration

## Runtime Note

Background execution is recommended for long jobs.

## Evaluation

- This test case is **manually evaluated**.
