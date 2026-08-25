# T282_ci_bids_validation_gate: CI Gate: BIDS Validation
## Task Description

Add a CI job that runs bids-validator on a small reference dataset and fails the build on errors.

## Input Requirement


- No interactive input.


## Constraints

- Reference dataset < 50 MB or stub-based.

- Exit-code handling documented.

- Save all generated artifacts to:
  - benchmark_results/T282_ci_bids_validation_gate/


## Expected Output

Expected output artifact(s):

- Workflow YAML

- `gate_behavior.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
