# T299_nvidia_container_toolkit_check: NVIDIA Container Toolkit Check
## Task Description

Verify GPU containers work: `docker run --gpus all` nvidia-smi, plus a PyTorch CUDA test inside the container, with troubleshooting notes.

## Input Requirement


- No interactive input.


## Constraints

- Document toolkit + driver versions.

- Include the common failure table (driver mismatch, no --gpus).

- Save all generated artifacts to:
  - benchmark_results/T299_nvidia_container_toolkit_check/


## Expected Output

Expected output artifact(s):

- `container_gpu_report.md`

- `troubleshooting_table.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
