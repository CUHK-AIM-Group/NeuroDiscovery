# Benchmark Test Case 31: FSL Tensor Fitting (DTIFIT)

## Task Description

Load corrected DWI and run tensor fitting.

## Input Requirement

Required input(s):

- Corrected DWI (required)
- bvecs/bvals (required)
- brain mask (required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use FSL-compatible workflow and commands.
- Save all generated artifacts to:
  - benchmark_results/T31_fsl_tensor_fit/
- Long-running processing is allowed to run as a background job.

## Expected Output

Expected output artifact(s):

- FA map
- MD map
- Principal diffusion direction outputs
- Optional additional DTIFIT outputs such as L1/L2/L3, tensor, S0, and mode are allowed but are not required to satisfy the benchmark.

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
