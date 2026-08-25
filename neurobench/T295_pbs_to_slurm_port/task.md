# T295_pbs_to_slurm_port: Port PBS Scripts to SLURM
## Task Description

Port existing PBS/Torque job scripts to SLURM equivalents, preserving resources and array semantics, with a side-by-side mapping table.

## Input Requirement


- No interactive input.


## Constraints

- Every PBS directive mapped or explicitly dropped with reason.

- Semantics (array indexing, env vars) verified.

- Save all generated artifacts to:
  - benchmark_results/T295_pbs_to_slurm_port/


## Expected Output

Expected output artifact(s):

- Ported .slurm scripts

- `pbs_slurm_mapping.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
