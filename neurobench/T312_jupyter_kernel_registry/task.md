# T312_jupyter_kernel_registry: Jupyter Kernel Registry
## Task Description

Register project conda environments as named Jupyter kernels with display names, and verify each kernel starts and imports its key package.

## Input Requirement


- No interactive input.


## Constraints

- Kernel names match env names.

- Startup test per kernel logged.

- Save all generated artifacts to:
  - benchmark_results/T312_jupyter_kernel_registry/


## Expected Output

Expected output artifact(s):

- `kernel_install_log.txt`

- `kernel_test.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
