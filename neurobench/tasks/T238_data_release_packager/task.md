# T238_data_release_packager: Data Release Packager
## Task Description

Package a dataset release: staging area, file manifest with SHA256, versioned tarball(s), and a RELEASE_NOTES file.

## Input Requirement

Required input(s):

- Dataset directory (required)

- Config/metadata inputs as stated (required)


If any required input is missing, return:

- Missing required input


## Constraints

- Tarballs split by modality if > 10 GB; state the rule used.

- Manifest covers every packaged file.

- Save all generated artifacts to:
  - benchmark_results/T238_data_release_packager/


## Expected Output

Expected output artifact(s):

- `release/` tree

- `MANIFEST.sha256`

- `RELEASE_NOTES.md`


Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json


## Evaluation

- This test case is manually evaluated.
