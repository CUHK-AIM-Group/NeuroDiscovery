# T294_slurm_dependency_chain: SLURM Dependency Chain
## Task Description

Compose a 3-stage SLURM workflow with dependencies: preprocessing -> denoising -> statistics, each stage starting only after the previous completes successfully.

## Input Requirement


- No interactive input.


## Constraints

- Use `--dependency=afterok:`.

- Each stage idempotent; document re-run behavior.

- Save all generated artifacts to:
  - benchmark_results/T294_slurm_dependency_chain/


## Expected Output

Expected output artifact(s):

- Three .slurm scripts

- `submit_chain.sh`

- `dag.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
