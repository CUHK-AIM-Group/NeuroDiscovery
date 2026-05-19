# Benchmark Test Case 33: FSL Tract Extraction (XTRACT)

## Task Description

Load local BEDPOSTX output and run XTRACT to extract major white matter tracts.

## Input Requirement

Required input(s):

- BEDPOSTX output folder (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T35_fsl_tract_extract/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Major white matter tract segmentation outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
