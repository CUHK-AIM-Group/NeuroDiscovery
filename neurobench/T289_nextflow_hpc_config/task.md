# T289_nextflow_hpc_config: Nextflow HPC Executor Config
## Task Description

Write a Nextflow config for an institutional SLURM/PBS cluster: queue selection, per-process resources, singularity enabled.

## Input Requirement


- No interactive input.


## Constraints

- Document queue names/partitions as comments.

- Test with `nextflow run -profile` hello-level pipeline.

- Save all generated artifacts to:
  - benchmark_results/T289_nextflow_hpc_config/


## Expected Output

Expected output artifact(s):

- `nextflow.config`

- `test_run_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
