# Benchmark Test Case 54: Nilearn End-to-end Pipeline

## Task Description

Load local preprocessed BOLD + confounds + atlas/events and run end-to-end Nilearn analysis pipeline (ROI + connectivity + GLM).

## Input Requirement

Required input(s):

- Preprocessed BOLD image (required)
- Confounds file (required)
- Atlas/parcellation and/or events inputs (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use Nilearn-compatible workflow and functions.
- Save all generated artifacts to:
  - benchmark_results/T77_nilearn_full_pipeline/

## Expected Output

Expected output artifact(s):

- All Nilearn derived outputs (ROI, connectome, GLM maps)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
