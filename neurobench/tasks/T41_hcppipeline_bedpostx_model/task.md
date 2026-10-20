# Benchmark Test Case 41: HCP Bedpostx Multi-fiber Modeling

## Task Description

Load local diffusion_preproc outputs and run bedpostx multi-fiber modeling.

## Input Requirement

Required input(s):

- Diffusion preprocessing output directory (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use HCP-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T56_hcp_bedpostx_model/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- Bedpostx multi-fiber model outputs

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
