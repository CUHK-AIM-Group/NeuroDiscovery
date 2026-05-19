# Benchmark Test Case 27: FSL Task GLM (FEAT)

## Task Description

Load corrected BOLD and design.fsf, run task fMRI GLM analysis with FEAT.

## Input Requirement

Required input(s):

- Motion-corrected BOLD (required)
- design.fsf (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T29_fsl_task_glm/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Statistical maps
- cope outputs
- zstat outputs
- FEAT result folder

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
