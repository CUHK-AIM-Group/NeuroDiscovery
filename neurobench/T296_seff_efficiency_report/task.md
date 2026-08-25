# T296_seff_efficiency_report: SLURM seff Efficiency Report
## Task Description

Analyze completed jobs with `seff`/`sacct`: CPU efficiency, memory efficiency, and walltime accuracy, then recommend right-sized requests per job type.

## Input Requirement


- No interactive input.


## Constraints

- Group by job name; percentiles reported.

- Recommendations in a table (old vs. new request).

- Save all generated artifacts to:
  - benchmark_results/T296_seff_efficiency_report/


## Expected Output

Expected output artifact(s):

- `efficiency_report.md`

- `sacct_dump.csv`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
