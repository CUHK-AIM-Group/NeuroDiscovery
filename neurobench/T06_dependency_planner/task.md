# Benchmark Test Case 6: Dependency Planner (PyTorch Compatibility)

## Task Description

Create a **new conda environment**, then install the **oldest PyTorch version** that the current machine environment can support.

## Input Requirement

- No interactive input.
- The agent should determine compatible Python/PyTorch/CUDA combinations by itself.

## Constraints

- Must create a brand-new conda environment (do not reuse existing envs).
- Must install PyTorch in that new environment.
- Prioritize the oldest compatible PyTorch version for this machine/runtime.
- Save execution result JSON to:
  - `benchmark_results/T06_dependency_planner/`

## Expected Output (JSON)

The agent should output one JSON file named like:
- `result_YYYYMMDD_HHMMSS.json`

Recommended structure:

```json
{
  "metadata": {
    "task": "T06_dependency_planner",
    "timestamp": "ISO-8601",
    "status": "success or fail"
  },
  "environment": {
    "name": "string",
    "python_version": "string",
    "conda_available": true
  },
  "pytorch": {
    "version": "string",
    "cuda_compiled": "string or null",
    "cuda_available": true,
    "smoke_test_passed": true
  },
  "system": {
    "gpu_present": true,
    "gpu_check_method": "nvidia-smi or torch"
  },
  "notes": "optional string"
}
```

## Success Criteria

- A new conda env is created successfully
- PyTorch can be imported in the new env
- Basic tensor compute smoke test passes
- If GPU is present on host, PyTorch can access GPU (`torch.cuda.is_available() == True`)
- Result JSON is written to `benchmark_results/T06_dependency_planner/`

## Verification Suggestion

You should use commands to self-verify during execution, for example:

- `conda env list`
- `conda run -n <env_name> python -c "import torch; print(torch.__version__)"`
- `conda run -n <env_name> python -c "import torch; print(torch.cuda.is_available())"`
- `nvidia-smi` (if available)
