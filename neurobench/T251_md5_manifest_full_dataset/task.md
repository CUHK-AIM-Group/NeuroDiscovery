# T251_md5_manifest_full_dataset: Full-Dataset Checksum Manifest
## Task Description

Generate a SHA256 (or MD5) manifest for every file in a dataset and a verification script that can re-check the manifest later.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Remote/credentials where applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Manifest sorted by path; one line per file.

- Verification script standalone (single python/bash file).

- Save all generated artifacts to:
  - benchmark_results/T251_md5_manifest_full_dataset/


## Expected Output

Expected output artifact(s):

- `MANIFEST.sha256`

- `verify_manifest.py` (or .sh)

- `generation_log.txt`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
