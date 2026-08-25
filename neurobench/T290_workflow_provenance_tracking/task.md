# T290_workflow_provenance_tracking: Workflow Provenance Tracking
## Task Description

Add provenance capture to a workflow: record tool versions, input hashes, and parameters per run in W3C-PROV-inspired JSON.

## Input Requirement


- No interactive input.


## Constraints

- One `prov.json` per run directory.

- Hashes of inputs included.

- Save all generated artifacts to:
  - benchmark_results/T290_workflow_provenance_tracking/


## Expected Output

Expected output artifact(s):

- `prov.json` example

- `provenance_reader.py`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
