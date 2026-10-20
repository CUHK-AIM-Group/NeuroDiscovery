# Benchmark Test Case 3: Conda Env Manager (Install htop)

## Task Description

Create a **new conda environment**, then install **htop**.

## Input Requirement

- No interactive input.

## Constraints

- Must create a brand-new conda environment (do not reuse existing envs).
- Must install `htop` in that new environment.
- Save execution result JSON to:
  - `benchmark_results/T03_conda_env_manager/`

## Expected Output (JSON)

The agent should output one JSON file named like:
- `result_YYYYMMDD_HHMMSS.json`

Recommended structure:

```json
{
  "metadata": {
    "task": "T03_conda_env_manager",
    "timestamp": "ISO-8601",
    "status": "success or fail"
  },
  "environment": {
    "name": "string",
    "python_version": "string",
    "conda_available": true
  },
  "htop": {
    "installed": true,
    "version": "string",
    "smoke_test_passed": true
  },
  "notes": "optional string"
}
```

## Success Criteria

- A new conda env is created successfully
- `htop` is installed in that env
- `htop --version` runs successfully in that env
- Result JSON is written to `benchmark_results/T03_conda_env_manager/`

## Verification Suggestion

Use commands to self-verify during execution, for example:

- `conda env list`
- `conda run -n <env_name> htop --version`
- `conda run -n <env_name> which htop`
