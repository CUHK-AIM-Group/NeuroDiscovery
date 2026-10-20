# T298_cuda_pytorch_smoke: CUDA + PyTorch Smoke Test
## Task Description

Verify the GPU stack end-to-end: nvidia-smi, CUDA version vs. PyTorch build compatibility, tensor op on GPU, and a 1-epoch micro training run.

## Input Requirement


- No interactive input.


## Constraints

- Report torch.version.cuda vs. nvidia-smi CUDA.

- Micro training must show decreasing loss (5 steps).

- Save all generated artifacts to:
  - benchmark_results/T298_cuda_pytorch_smoke/


## Expected Output

Expected output artifact(s):

- `gpu_smoke_report.json`

- `micro_train_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
