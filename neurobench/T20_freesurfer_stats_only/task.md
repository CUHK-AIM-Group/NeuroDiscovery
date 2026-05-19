# Benchmark Test Case 20: FreeSurfer Stats-only Export

## Task Description

Use an already reconstructed FreeSurfer subject (T1w-based) and generate only statistics outputs.

Goal:

- avoid full reconstruction rerun
- generate all `.stats` files under `stats/`

## Input Requirement

Required:

- an existing FreeSurfer subject directory with completed reconstruction

If required input is missing, return:

- `Missing required input`

## Expected Workflow

1. Locate completed subject
2. Run stats-only commands/workflow
3. Collect all generated/available `.stats` files

## Output Requirement

Save artifacts to:

- `benchmark_results/T20_freesurfer_stats_only/`

Expected key output:

- all `.stats` files under `stats/` directory copied/exported to result artifact tree

## Runtime Note

Can run in background if needed.

## Evaluation

- This test case is **manually evaluated**.
