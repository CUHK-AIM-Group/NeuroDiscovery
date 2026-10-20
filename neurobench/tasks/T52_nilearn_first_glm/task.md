# Benchmark Test Case 52: Nilearn First-level GLM

## Task Description

Load local preprocessed BOLD + events + confounds, run first-level GLM analysis, and output statistical maps.

## Input Requirement

Required input(s):

- Preprocessed BOLD image (required)
- Events file (required)
- Confounds file (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use Nilearn-compatible workflow and functions.
- Save all generated artifacts to:
  - benchmark_results/T75_nilearn_first_glm/

## Expected Output

Expected output artifact(s):

- first_level_zmap.nii.gz
- COPE output(s)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
