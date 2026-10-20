# T324_freesurfer_license_mgmt: FreeSurfer License Management
## Task Description

Document and script correct FreeSurfer license handling across local/Docker/Singularity runs: where the file lives, how it is mounted, and how CI handles it.

## Input Requirement


- No interactive input.


## Constraints

- Never commit the license file.

- Mount examples per runtime.

- Save all generated artifacts to:
  - benchmark_results/T324_freesurfer_license_mgmt/


## Expected Output

Expected output artifact(s):

- `FREESURFER_LICENSE.md`

- `mount_examples.sh`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
