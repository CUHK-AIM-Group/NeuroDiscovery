# T325_phi_scan_git_history: PHI Scan of Git History
## Task Description

Scan the repository git history for accidentally committed subject identifiers or PHI patterns, and produce a remediation plan (BFG/filter-repo) if found.

## Input Requirement


- No interactive input.


## Constraints

- Pattern list documented.

- No history rewriting in this task; plan only.

- Save all generated artifacts to:
  - benchmark_results/T325_phi_scan_git_history/


## Expected Output

Expected output artifact(s):

- `phi_history_scan.json`

- `remediation_plan.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
