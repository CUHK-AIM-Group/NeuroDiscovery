# Benchmark Test Case 37: HCP Functional Volume Pipeline

## Task Description

Load local BOLD data (task or resting-state) and run HCP fMRIVolume preprocessing.

## Input Requirement

Required input(s):

- Local BOLD image (task or resting-state, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T52_hcp_functional_volume/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- fMRIVolume preprocessed outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
