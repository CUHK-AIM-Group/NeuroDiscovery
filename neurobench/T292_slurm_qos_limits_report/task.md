# T292_slurm_qos_limits_report: SLURM QoS / Limits Report
## Task Description

Inventory the cluster's SLURM limits relevant to neuroimaging jobs: partitions, QoS, max walltime, memory per node, and produce a job-sizing cheat sheet.

## Input Requirement


- No interactive input.


## Constraints

- Use `sacctmgr`/`sinfo` read-only commands.

- Cheat sheet as Markdown table.

- Save all generated artifacts to:
  - benchmark_results/T292_slurm_qos_limits_report/


## Expected Output

Expected output artifact(s):

- `slurm_limits.md`

- `sinfo_snapshot.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
