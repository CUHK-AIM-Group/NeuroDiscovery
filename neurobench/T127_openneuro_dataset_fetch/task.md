# T127_openneuro_dataset_fetch: OpenNeuro Dataset Fetch and Integrity Check
## Task Description

Fetch one OpenNeuro dataset (default `ds000114`, configurable) via DataLad or
the S3 mirror, materialize the required files, and verify integrity against
the dataset manifest.

## Input Requirement

Required input(s):

- OpenNeuro dataset accession (e.g. `ds000114`, required)

If any required input is missing, return:

- Missing required input

## Constraints

- Use `datalad install` + `datalad get`, or `aws s3 sync --no-sign-request`.
- Do not download more than the requested modalities (document what was
  materialized vs. left as symlinks/annexed).
- Verify file count and, where available, checksums.
- Save all generated artifacts to:
  - benchmark_results/T127_openneuro_dataset_fetch/

## Expected Output

Expected output artifact(s):

- Local dataset clone with materialized files
- `integrity_report.json` (expected vs. actual file counts, failures)
- `dataset_card.md` (short summary: subjects, tasks, modalities, license)

Recommended metadata file:

- result_YYYYMMDD_HHMMSS.json

## Evaluation

- This test case is manually evaluated.
