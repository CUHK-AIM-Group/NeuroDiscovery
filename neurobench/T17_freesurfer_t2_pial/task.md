# Benchmark Test Case 17: FreeSurfer T2-assisted Pial Refinement

## Task Description

Load local T1w + T2w and run FreeSurfer with T2-assisted pial surface refinement.

Goal:

- improve pial surface precision compared to T1-only baseline

## Input Requirement

Required:

- local T1w image
- local T2w image

If required input is missing, return:

- `Missing required input`

## Expected Workflow

1. Load T1w and T2w
2. Run FreeSurfer with T2 pial refinement option
3. Export refined pial surfaces and logs

## Output Requirement

Save artifacts to:

- `benchmark_results/T17_freesurfer_t2_pial/`

Expected key outputs:

- refined pial surfaces (`surf/lh.pial`, `surf/rh.pial`)
- associated segmentation/statistics outputs

## Runtime Note

Long runtime; background execution is acceptable.

## Evaluation

- This test case is **manually evaluated**.
