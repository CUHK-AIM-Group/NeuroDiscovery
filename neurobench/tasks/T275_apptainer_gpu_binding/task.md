# T275_apptainer_gpu_binding: Apptainer GPU Binding Test
## Task Description

Verify GPU passthrough into an Apptainer container: nvidia-smi inside the container, PyTorch CUDA availability test, and driver/runtime compatibility notes.

## Input Requirement


- No interactive input.


## Constraints

- Use `--nv`; document host driver version.

- Report CUDA devices visible inside the container.

- Save all generated artifacts to:
  - benchmark_results/T275_apptainer_gpu_binding/


## Expected Output

Expected output artifact(s):

- `gpu_binding_report.md`

- `torch_cuda_test.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
