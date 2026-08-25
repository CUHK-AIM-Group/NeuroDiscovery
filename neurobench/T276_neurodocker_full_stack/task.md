# T276_neurodocker_full_stack: Neurodocker Full-Stack Image
## Task Description

Generate a Dockerfile with neurodocker containing FSL + AFNI + ANTs + MRtrix3 + converted dcm2niix, build it, and smoke-test every tool.

## Input Requirement


- No interactive input.


## Constraints

- Neurodocker command kept; versions pinned.

- Smoke test per tool (version command).

- Save all generated artifacts to:
  - benchmark_results/T276_neurodocker_full_stack/


## Expected Output

Expected output artifact(s):

- `Dockerfile`

- `neurodocker_command.txt`

- `smoke_test.json`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
