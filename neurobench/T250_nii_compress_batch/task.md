# T250_nii_compress_batch: Batch nii -> nii.gz Compression
## Task Description

Compress uncompressed NIfTIs in a dataset to .nii.gz, updating BIDS references (scans.tsv, IntendedFor) and verifying data integrity.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Remote/credentials where applicable (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Integrity: voxel data identical after round-trip (verify via checksum of decompressed stream).

- Update every referencing sidecar.

- Save all generated artifacts to:
  - benchmark_results/T250_nii_compress_batch/


## Expected Output

Expected output artifact(s):

- Compressed files

- `compression_log.csv`

- `reference_updates.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
