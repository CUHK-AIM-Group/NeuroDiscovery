# T308_repro_env_capture: Full Environment Capture Report
## Task Description

Capture the complete computational environment of an analysis machine: OS, kernel, Python/R/conda packages, container versions, GPU stack, into one reproducibility report.

## Input Requirement


- No interactive input.


## Constraints

- Single script generates everything.

- Report includes capture timestamp + hostname.

- Save all generated artifacts to:
  - benchmark_results/T308_repro_env_capture/


## Expected Output

Expected output artifact(s):

- `env_capture.sh`

- `environment_report.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
