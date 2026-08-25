# T274_singularity_pull_convert: Singularity/Apptainer Image Conversion
## Task Description

Pull a BIDS-app Docker image and convert it to a Singularity/Apptainer SIF for HPC use, then verify execution.

## Input Requirement


- No interactive input.


## Constraints

- Pin the Docker tag by digest.

- Run `--version` inside the SIF as verification.

- Save all generated artifacts to:
  - benchmark_results/T274_singularity_pull_convert/


## Expected Output

Expected output artifact(s):

- `image.sif` reference (or pull command record)

- `conversion_log.txt`

- `sif_exec_test.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
